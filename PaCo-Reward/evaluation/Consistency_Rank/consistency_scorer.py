import os
import re
import json
from typing import List, Tuple, Union, Optional, Any
from io import BytesIO
import base64
import logging
import asyncio
from itertools import combinations
import hashlib
import math
import time

import torch
import numpy as np
import openai
from openai import OpenAI, AsyncOpenAI
from PIL import Image

# VLLM log filter
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

def hash_pil_image(image: Image.Image) -> str:
    """
        Generate a hash string for a PIL Image.
        Args:
            image (Image.Image): PIL Image object
        Returns:
            str: Hash string of the image
    """
    return hashlib.md5(image.tobytes()).hexdigest()

def pil_image_to_base64(image, format="JPEG"):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image/{format.lower()};base64,{encoded_image_text}"
    return base64_qwen

def divide_prompt(prompt):
    # seqis like ". [TOP-LEFT]:"
    match_sep = re.compile(r"\.\s+[A-Z0-9-\[\]]+:")
    seps = match_sep.findall(prompt)
    # Add '.' for each sentence
    sub_prompts = [
        p + '.' if p.strip()[-1] != '.' else p
        for p in re.split('|'.join(map(re.escape, seps)), prompt)
    ]
    return sub_prompts

def divide_image(image, grid_info : tuple[int, int]):
    assert len(grid_info) == 2, "grid_info must be a tuple of two integers (a, b)"

    a, b = grid_info
    width, height = image.size

    grid_cells = []
    cell_width = width // b
    cell_height = height // a

    # 2x2 grid
    # | 1 | 2 |
    # | 3 | 4 |
    # [
    # (0, 0, cell_width, cell_height),
    # (cell_width, 0, 2 * cell_width, cell_height),
    # (0, cell_height, cell_width, 2 * cell_height),
    # (cell_width, cell_height, 2 * cell_width, 2 * cell_height)
    # ]

    for i in range(a):
        for j in range(b):
            upper = i * cell_height
            left = j * cell_width
            right = left + cell_width
            lower = upper + cell_height
            grid_cells.append(image.crop((left, upper, right, lower)))

    return grid_cells

def extract_grid_info(prompt) -> tuple[int, int]:
    # Grid can be represented as int x int, or int ⨉ int. ⨉ has unicode \u2a09
    match = re.findall(r'(\d+)\s*[x⨉]\s*(\d+)', prompt)
    if len(match) == 0:
        return (1, 1)

    return (int(match[0][0]), int(match[0][1]))


def get_yes_prob_from_completion(completion : openai.ChatCompletion) -> float:
    if completion is None:
        return 0.0

    logprobs = completion.choices[0].logprobs
    if logprobs:
        # Use logprobs to compute, score = P('yes') / (P('yes') + P('no'))
        # score = 1 / (1 + exp(logprob('no') -  logprob('yes')))
        # Same formular for logits as well. Since the sum term will cancel out.
        # Use uppercase only here.
        token_logprobs = {t.token: t.logprob for t in logprobs.content[0].top_logprobs}
        yes_logprob = token_logprobs.get('Yes', float('-inf'))
        no_logprob = token_logprobs.get('No', float('-inf'))

        if yes_logprob == float('-inf') and no_logprob == float('-inf'):
            # When inf - inf encountered, give 0.0 score.
            score = 0.0 # 0.0
        else:
            diff = torch.tensor(yes_logprob - no_logprob, dtype=torch.float64)
            score = torch.sigmoid(diff).item()
    else:
        score = 0.0

    return score


class ConsistencyScorer:
    def __init__(
            self,
            api_key: str = 'dummy-key',
            base_url: str = 'http://127.0.0.1:8000/v1',
            model='QwenVL2.5-7B-Instruct',
            async_mode=True,
            max_concurrent=12,  # 2x2 grid has 6 pair of images to compare. 12 for at most 2 batches at once.
            max_retries=10,
            timeout=60,
            thinking=False,
            prompt_template: int = 1,
            max_cache_size=1024,
        ):
        self.model = model
        self.async_mode = async_mode
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout = timeout
        self.thinking = thinking
        self.prompt_template = prompt_template
        self.max_cache_size = max_cache_size if max_cache_size is not None else math.inf

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.cache : dict[tuple[str, str, str], float] = {} # (img1_hash, img2_hash, criteria_text) -> score

    def add_to_cache(self, key: tuple[str, str, str], value: float):
        if len(self.cache) >= self.max_cache_size:
            # Remove the oldest item in the cache (FIFO)
            self.cache.pop(next(iter(self.cache)))

        self.cache[key] = value

    @torch.no_grad()
    async def __call__(self, ref_images : list[Image.Image], images : list[Image.Image], metadatas : list[dict]) -> list[float]:
        # Create a global semaphore for overall concurrency control
        global_semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_single_image(ref_image, image, metadata):
            async with global_semaphore:
                if self.prompt_template == 1:
                    criteria_info = metadata['criteria']
                else:
                    criteria_info = {
                        "dummy-dimension": {"dummy-criterion": "dummy-text"},
                    }
                prompt = metadata.get('prompt', None)
                dimensions = criteria_info.keys()
                dimension_scores = {k:0.0 for k in dimensions}
                
                # Compute scores for each prompt-image pair from different dimensions
                for dimension in dimensions:
                    # Get criteria for this dimension
                    dimension_criteria = criteria_info[dimension]
                    criteria_texts = [c_t for c_t in dimension_criteria.values() if c_t]

                    # Compute scores for each criterion
                    criterion_scores = []
                    for ct in criteria_texts:
                        score = await self.compute_image_consistency(ref_image, image, ct, prompt)
                        criterion_scores.append(score)

                    # # Compute the average score within each criterion
                    criterion_scores = np.mean(criterion_scores) if criterion_scores else 0.0

                    # # Compute the overall score for this dimension
                    dimension_scores[dimension] = criterion_scores

                # Compute average scores from each dimension
                return sum(dimension_scores.values()) / len(dimension_scores)

        # Process all images concurrently
        tasks = [
            process_single_image(ref_image, image, metadata) 
            for ref_image, image, metadata in zip(ref_images, images, metadatas)
        ]
        
        final_scores = await asyncio.gather(*tasks)
        return final_scores
    
    async def compute_image_consistency(
            self,
            ref_image : Image.Image,
            image : Image.Image,
            criteria_text : str,
            prompt: Optional[str] = None,
            top_logprobs: int = 20
        ) -> list[float]:
        return await self._async_compute_image_consistency(ref_image, image, criteria_text, prompt, top_logprobs)

    async def _async_compute_image_consistency(
            self,
            ref_image : Image.Image,
            image : Image.Image,
            criteria_text : str,
            prompt: Optional[str] = None,
            top_logprobs: int = 20
        ) -> float:
        """
        Async version of compute_image_consistency with concurrency control.
        """
        # # v1 prompt
        if self.prompt_template == 1:
            hash_key = (hash_pil_image(ref_image), hash_pil_image(image), criteria_text)
            if hash_key in self.cache:
                return self.cache[hash_key]
            text_prompt = f"Do images meet the following criteria? {criteria_text} Please answer Yes or No." 
        elif self.prompt_template == 2:
            # v2 prompt
            hash_key = (hash_pil_image(ref_image), hash_pil_image(image), "")
            if hash_key in self.cache:
                return self.cache[hash_key]
            text_prompt = f"Do the two images maintain consistency in terms of style, logic and identity? Answer \"Yes\" and \"No\" only."
        elif self.prompt_template == 3:
            # v3 prompt
            hash_key = (hash_pil_image(ref_image), hash_pil_image(image), "")
            if hash_key in self.cache:
                return self.cache[hash_key]
            text_prompt = f"Do the two images maintain consistency in terms of style, logic and identity? Answer \"Yes\" and \"No\" first, and then provide detailed reasons."
        messages = [
            {
                "role": "user",
                "content":
                [
                    {"type": "image_url", "image_url": {"url": pil_image_to_base64(ref_image)}},
                    {"type": "image_url", "image_url": {"url": pil_image_to_base64(image)}},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]
        for attempt in range(self.max_retries):
            try:
                if not self.thinking:
                    completion = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0, # Deterministic result, no use for logprobs, actually.
                        max_completion_tokens=1,
                        logprobs=True,
                        top_logprobs=top_logprobs,
                        timeout=self.timeout
                    )
                    break
                else:
                    # Thinking mode, stream the response to find the first token after '<answer>'
                    stream = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0, # Deterministic result, no use for logprobs
                        logprobs=True,
                        top_logprobs=top_logprobs,
                        stream=True,
                    )
                    async for chunk in stream:
                        # Skip thing process to reach the first token after <answer>
                        if chunk.choices[0].delta.content.strip() == '<answer>':
                            first_token = await anext(stream, None)
                            if first_token and first_token.choices[0].delta.content == '<|begin_of_box|>':
                                first_token = await anext(stream, None)
                            completion = first_token
                            break
                    break
            except Exception as e:
                print(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise e
        
        score = get_yes_prob_from_completion(completion)
        self.add_to_cache(hash_key, score)
        return score