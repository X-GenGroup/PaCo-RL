
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
from ..utils import get_yes_cond_prob_from_completion, pil_image_to_base64, hash_pil_image

# VLLM log filter
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
class ConsistencyScorerForEditing:
    """
        This class tasks a list of combined images (a grid layout image) and a list of corresponding prompts to compute consistency scores.
    """
    def __init__(
            self,
            client: AsyncOpenAI,
            model='ConsistencyReward-7B',
            max_concurrent=100,
            max_retries=10,
            timeout=60,
            max_cache_size=1024,
        ):
        self.client = client
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout = timeout
        self.global_semaphore = asyncio.Semaphore(self.max_concurrent)
        self.max_cache_size = max_cache_size if max_cache_size is not None else math.inf
        self.cache : dict[tuple[str, str, str], float] = {} # (img1_hash, img2_hash, criteria_text) -> score

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
            ref_image = metadata['ref_image']
            # consistency_text_prompt = f"Given two images after an edit with instruction '{prompt}', do they maintain consistency in terms of style, logic and identity? Answer 'Yes' or 'No' only."
            # prompt_following_text_prompt = f"Given two images after an edit with instruction '{prompt}', is the edit following the instruction well? Answer 'Yes' or 'No' only."
            consistency_text_prompt = (
                f"Compare the edited image (second) with the original image (first). "
                f"Instruction: '{prompt}'. "
                f"Except for the parts that are intentionally changed according to the instruction, "
                f"does the edited image remain consistent with the original in style, logic, and identity? "
                f"Answer 'Yes' or 'No' first, then provide detailed reasons."
            )
            prompt_following_text_prompt = (
                f"Compare the edited image (second) with the original image (first). "
                f"Instruction: '{prompt}'. "
                f"Does the edited image accurately follow this instruction? "
                f"Answer 'Yes' or 'No' first, then provide detailed reasons."
            )
            consistency_score = await self._async_compute_image_consistency(
                criteria_text=consistency_text_prompt,
                ref_image=ref_image,
                image=image,
                canonicalize=canonicalize
            )
            prompt_following_score = await self._async_compute_image_consistency(
                criteria_text=prompt_following_text_prompt,
                ref_image=ref_image,
                image=image,
                canonicalize=canonicalize
            )
            # Combine the two scores (e.g., geometric mean following EditScore)
            combined_score = math.sqrt(consistency_score * prompt_following_score)
            return combined_score
        
        tasks = [
            process_single_image(prompt, image, metadata)
            for prompt, image, metadata in zip(prompts, images, metadatas)
        ]
        scores = await asyncio.gather(*tasks)
        return scores

    async def _async_compute_image_consistency(
            self,
            criteria_text: str,
            ref_image: Image.Image,
            image: Image.Image,
            top_logprobs: int = 20,
            canonicalize: bool = False
        ) -> float:
        """
        Async version of compute_image_consistency with concurrency control.
        """
        # cache_key = (hash_pil_image(image), hash_pil_image(ref_image), criteria_text)
        # if cache_key in self.cache:
        #     return self.cache[cache_key]
        messages = [
            {
                "role": "user",
                "content":
                [
                    {"type": "image_url", "image_url": {"url": pil_image_to_base64(ref_image)}},
                    {"type": "image_url", "image_url": {"url": pil_image_to_base64(image)}},
                    {"type": "text", "text": criteria_text},
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
                # self.add_to_cache(cache_key, score)
                break

            except Exception as e:
                print(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    score = 0.0  # Default score on failure        
        return score    

