import os
import re
import json
from typing import List, Tuple, Union
from io import BytesIO
import base64
import logging
import asyncio
import time
from unittest import result
import math
import torch
import numpy as np
import openai
from openai import OpenAI, AsyncOpenAI
from PIL import Image
from ..utils import pil_image_to_base64, divide_image, extract_grid_info, hash_pil_image, get_yes_cond_prob_from_completion

# VLLM log filter
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


class GridLayoutScorer:
    def __init__(
            self,
            client : AsyncOpenAI,
            model='Qwen2.5-VL-7B-Instruct',
            max_concurrent=100,
            max_retries=10,
            timeout=60,
            max_cache_size=1024,
        ):
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout = timeout
        self.global_semaphore = asyncio.Semaphore(self.max_concurrent)
        self.max_cache_size = max_cache_size if max_cache_size is not None else math.inf
        self.cache : dict[tuple[str, tuple[int, int]], float] = {} # (img_hash, (rows, cols)) -> score

        self.client = client

    def __call__(self, images : list[Image.Image], prompts : list[str], metadatas : list[dict]) -> list[float]:
        return asyncio.run(self.__async_call__(images, prompts, metadatas))

    def add_to_cache(self, key: tuple[str, tuple[int, int]], value: float):
        if len(self.cache) >= self.max_cache_size:
            # Remove the oldest item in the cache (FIFO)
            self.cache.pop(next(iter(self.cache)))

        self.cache[key] = value
        return value

    @torch.no_grad()
    async def __async_call__(self, images : list[Image.Image], prompts : list[str], metadatas : list[dict]) -> list[float]:
        assert len(prompts) == len(images), "Length of prompts and images must match"

        # Process all images concurrently
        tasks = [
            self.compute_layout_score(prompt, image, metadata, use_prob=False)
            for prompt, image, metadata in zip(prompts, images, metadatas)
        ]

        final_scores = await asyncio.gather(*tasks)
        return final_scores
    
    async def compute_layout_score(
            self,
            prompt : str,
            image : Image.Image,
            metadata : dict,
            top_logprobs: int = 20,
            use_prob: bool = False
        ) -> float:
        img_hash = hash_pil_image(image)
        grid_info = extract_grid_info(prompt)
        cache_key = (img_hash, grid_info)
        if cache_key in self.cache:
            return self.cache[cache_key]
        messages = [
            {
                "role": "user",
                "content":
                [
                    {"type": "image_url", "image_url": {"url": pil_image_to_base64(image)}},
                    {"type": "text", "text": f"Is it a {grid_info} grid layout image? Please answer Yes or No."},
                ]
            }
        ]
        completion = None
        for attempt in range(self.max_retries):
            try:
                async with self.global_semaphore:
                    completion = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=1e-3, # temperature doesn't matter if we only look at the prob of "Yes"
                        max_completion_tokens=1,
                        logprobs=True,
                        top_logprobs=top_logprobs,
                        timeout=self.timeout
                    )
                    break
            except Exception as e:
                print(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        if completion is None:
            # Do not cache failed attempts
            return 0.0

        if use_prob:
            score = get_yes_cond_prob_from_completion(completion, canonicalize=True)
        else:
            content = completion.choices[0].message.content.strip().lower()
            score = 1.0 if "yes" in content else 0.0
        
        self.add_to_cache(cache_key, score)
        return score