import os
import re
import json
from socket import timeout
from typing import List, Tuple, Union
from io import BytesIO
import base64
import logging
import asyncio
from itertools import combinations
import math
import time

import torch
import numpy as np
import openai
from openai import OpenAI, AsyncOpenAI
from PIL import Image
from ..utils import pil_image_to_base64, divide_image, extract_grid_info, hash_pil_image, get_yes_cond_prob_from_completion, divide_prompt

# VLLM log filter
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
class ConsistencyScorer:
    """
        This class tasks a list of combined images (a grid layout image) and a list of corresponding prompts to compute consistency scores.
    """
    def __init__(
            self,
            client: AsyncOpenAI,
            model='Qwen2.5-VL-7B-Instruct',
            max_concurrent=100,
            max_retries=10,
            timeout=60,
            max_cache_size=1024,
            prompt_template_version: int = 1,
        ):
        self.client = client
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout = timeout
        self.global_semaphore = asyncio.Semaphore(self.max_concurrent)
        self.max_cache_size = max_cache_size if max_cache_size is not None else math.inf
        self.cache : dict[tuple[str, str, str], float] = {} # (img1_hash, img2_hash, criteria_text) -> score
        self.prompt_template_version = prompt_template_version

    def add_to_cache(self, key: tuple[str, str, str], value: float):
        if len(self.cache) >= self.max_cache_size:
            # Remove the oldest item in the cache (FIFO)
            self.cache.pop(next(iter(self.cache)))

        self.cache[key] = value

    def __call__(self, images : list[Image.Image], prompts : list[str], metadatas : list[dict], canonicalize: bool = False) -> list[float]:
        return asyncio.run(self.__async_call__(images, prompts, metadatas, canonicalize))

    @torch.no_grad()
    async def __async_call__(self, images : list[Image.Image], prompts : list[str], metadatas : list[dict], canonicalize: bool = False) -> list[float]:
        assert len(prompts) == len(images), "Length of prompts and images must match"

        async def process_single_image(prompt, image, metadata):
            criteria_info = metadata['criteria']
            dimensions = criteria_info.keys()
            dimension_scores = {k: 0.0 for k in dimensions}
            
            # Compute scores for each prompt-image pair from different dimensions
            for dimension in dimensions:
                # Get criteria for this dimension
                dimension_criteria = criteria_info[dimension]
                criteria_texts = [c_t for c_t in dimension_criteria.values() if c_t]

                # [criteria1_scores : list[float], criteria2_scores : list[float], ...]
                criterion_scores = []
                for ct in criteria_texts:
                    scores = await self._async_compute_image_consistency(prompt, image, ct, canonicalize=canonicalize)
                    criterion_scores.append(scores)

                # Compute the average score within each criterion
                criterion_scores = [sum(scores) / len(scores) if scores else 0.0 for scores in criterion_scores]

                # Compute the overall score for this dimension
                overall_score = sum(criterion_scores) / len(criterion_scores) if criterion_scores else 0.0
                dimension_scores[dimension] = overall_score

            # Compute average scores from each dimension
            return sum(dimension_scores.values()) / len(dimension_scores)

        # Process all images concurrently
        tasks = [
            process_single_image(prompt, image, metadata) 
            for prompt, image, metadata in zip(prompts, images, metadatas)
        ]
        
        final_scores = await asyncio.gather(*tasks)
        return final_scores

    async def _async_compute_image_consistency(
            self,
            prompt: str,
            image: Image.Image,
            criteria_text: str,
            top_logprobs: int = 20,
            canonicalize: bool = False
        ) -> list[float]:
        """
        Async version of compute_image_consistency with concurrency control.
        """
        if self.prompt_template_version == 0:
            text_prompt = f"Do images meet the following criteria? {criteria_text} Please answer Yes or No first, then provide detailed reasons."
        elif self.prompt_template_version == 1:
            sub_prompts = divide_prompt(prompt)
            main_prompt = sub_prompts[0]  # Use the main prompt for context if needed
            text_prompt = (
                f"Given two subfigures generated based on the theme: \"{main_prompt}\", "
                f"do the two images maintain consistency in terms of style, logic and identity? "
                f"Answer \"Yes\" and \"No\" first, and then provide detailed reasons."
            )
        elif self.prompt_template_version == 2:
            # Not good
            sub_prompts = divide_prompt(prompt)
            main_prompt = sub_prompts[0]  # Use the main prompt for context if needed
            text_prompt = (
                    f"Given two subfigures generated based on the theme: \"{main_prompt}\", "
                    f"do the two images maintain consistency in terms of style, logic, and identity? "
                    f"If the two images look almost identical or duplicated, answer \"No\". "
                    f"Otherwise, answer \"Yes\" or \"No\" first, and then provide detailed reasons."
                )
        else:
            text_prompt = f"Do images meet the following criteria? {criteria_text} Please answer Yes or No first, then provide detailed reasons."
        async def process_image_pair(image1, image2) -> float:
            cache_key = (hash_pil_image(image1), hash_pil_image(image2), criteria_text)
            if cache_key in self.cache:
                return self.cache[cache_key]
            messages = [
                {
                    "role": "user",
                    "content":
                    [
                        {"type": "image_url", "image_url": {"url": pil_image_to_base64(image1)}},
                        {"type": "image_url", "image_url": {"url": pil_image_to_base64(image2)}},
                        {"type": "text", "text": text_prompt},
                    ]
                }
            ]
            for attempt in range(self.max_retries):
                try:
                    async with self.global_semaphore:
                        completion = await self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=0.0, # Deterministic result, no use for logprobs, actually.
                            max_completion_tokens=1,
                            logprobs=True,
                            top_logprobs=top_logprobs,
                            timeout=self.timeout
                        )

                    score = get_yes_cond_prob_from_completion(completion, canonicalize=canonicalize)
                    self.add_to_cache(cache_key, score)
                    return score

                except Exception as e:
                    print(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        return 0.0  # Return a default score on failure, do not cache failed attempts

        grid_info = extract_grid_info(prompt)
        sub_images = divide_image(image, grid_info)
        
        # Create tasks for all image pairs (no additional semaphore here since global control is in __call__)
        tasks = []
        for image1, image2 in combinations(sub_images, 2):
            task = process_image_pair(image1, image2)
            tasks.append(task)

        # Execute all tasks concurrently
        scores = await asyncio.gather(*tasks)

        return scores