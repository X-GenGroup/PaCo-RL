import torch
import numpy as np
import hashlib
import math
from typing import List, Optional
from PIL import Image

from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


def hash_pil_image(image: Image.Image) -> str:
    """Generate hash for PIL Image."""
    return hashlib.md5(image.tobytes()).hexdigest()


def apply_chat_template(prompt: str, num_images: int = 2) -> str:
    """Apply chat template for Qwen2.5-VL."""
    template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    template += "".join([f"<img{i}>: <|vision_start|><|image_pad|><|vision_end|>" for i in range(1, num_images + 1)])
    template += f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return template


class ConsistencyScorerVLLM:
    def __init__(
        self,
        model: str = 'QwenVL2.5-7B-Instruct',
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        prompt_template: int = 1,
        max_cache_size: int = 1024,
    ):
        self.model_name = model
        self.prompt_template = prompt_template
        self.max_cache_size = max_cache_size if max_cache_size is not None else math.inf
        
        # Initialize VLLM model
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 2},
            enable_prefix_caching=True,
        )
        
        # Sampling params to get logprobs
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20,  # Get top 20 token logprobs
        )
        
        self.cache: dict[tuple[str, str, str], float] = {}

    def add_to_cache(self, key: tuple[str, str, str], value: float):
        if len(self.cache) >= self.max_cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value

    def prepare_input(self, images: List[Image.Image], text_prompt: str):
        """Prepare input in VLLM format."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": text_prompt}],
            }
        ]
        
        text = apply_chat_template(text_prompt, num_images=len(images))
        image_inputs, _ = process_vision_info(messages)
        
        return {
            "prompt": text,
            "multi_modal_data": {"image": image_inputs},
        }

    @torch.no_grad()
    def __call__(
        self,
        ref_images: List[Image.Image],
        images: List[Image.Image],
        metadatas: List[dict]
    ) -> List[float]:
        """Compute consistency scores for multiple image pairs."""
        final_scores = []
        
        for ref_image, image, metadata in zip(ref_images, images, metadatas):
            if self.prompt_template == 1:
                criteria_info = metadata['criteria']
            else:
                criteria_info = {
                    "dummy-dimension": {"dummy-criterion": "dummy-text"},
                }
            prompt = metadata.get('prompt', None)
            dimensions = criteria_info.keys()
            dimension_scores = {k: 0.0 for k in dimensions}
            
            # Compute scores for each dimension
            for dimension in dimensions:
                dimension_criteria = criteria_info[dimension]
                criteria_texts = [c_t for c_t in dimension_criteria.values() if c_t]
                
                criterion_scores = []
                for ct in criteria_texts:
                    score = self.compute_image_consistency(ref_image, image, ct, prompt)
                    criterion_scores.append(score)
                
                dimension_scores[dimension] = np.mean(criterion_scores) if criterion_scores else 0.0
            
            final_score = sum(dimension_scores.values()) / len(dimension_scores)
            final_scores.append(final_score)
        
        return final_scores

    def compute_image_consistency(
        self,
        ref_image: Image.Image,
        image: Image.Image,
        criteria_text: str,
        prompt: Optional[str] = None,
    ) -> float:
        """Compute consistency score between two images."""
        # Check cache
        if self.prompt_template == 1:
            hash_key = (hash_pil_image(ref_image), hash_pil_image(image), criteria_text)
            text_prompt = f"Do images meet the following criteria? {criteria_text} Please answer Yes or No."
        elif self.prompt_template == 2:
            hash_key = (hash_pil_image(ref_image), hash_pil_image(image), "")
            text_prompt = "Do the two images maintain consistency in terms of style, logic and identity? Answer \"Yes\" and \"No\" only."
        elif self.prompt_template == 3:
            # v3 prompt
            hash_key = (hash_pil_image(ref_image), hash_pil_image(image), "")
            if hash_key in self.cache:
                return self.cache[hash_key]
            text_prompt = f"Do the two images maintain consistency in terms of style, logic and identity? Answer \"Yes\" and \"No\" first, and then provide detailed reasons."
            
        if hash_key in self.cache:
            return self.cache[hash_key]
        
        # Prepare input
        input_data = self.prepare_input([ref_image, image], text_prompt)
        
        # Generate with VLLM
        outputs = self.llm.generate(
            input_data,
            sampling_params=self.sampling_params,
            use_tqdm=False
        )
        
        # Extract score from output
        score = self._extract_yes_probability(outputs[0])
        
        self.add_to_cache(hash_key, score)
        return score

    def _extract_yes_probability(self, output) -> float:
        """Extract P(Yes) / (P(Yes) + P(No)) from VLLM output."""
        if not output.outputs or not output.outputs[0].logprobs:
            return 0.0
        
        # Get logprobs for first token
        first_token_logprobs = output.outputs[0].logprobs[0]
        
        # Extract Yes/No logprobs
        yes_logprob = float('-inf')
        no_logprob = float('-inf')
        
        for token_id, logprob_obj in first_token_logprobs.items():
            token_str = logprob_obj.decoded_token
            if token_str == "Yes":
                yes_logprob = logprob_obj.logprob
            elif token_str == 'No':
                no_logprob = logprob_obj.logprob
        
        # Compute probability: sigmoid(yes_logprob - no_logprob)
        if yes_logprob == float('-inf') and no_logprob == float('-inf'):
            return 0.0
        
        diff = torch.tensor(yes_logprob - no_logprob, dtype=torch.float64)
        score = torch.sigmoid(diff).item()
        
        return score

    def batch_compute(
        self,
        ref_images: List[Image.Image],
        images: List[Image.Image],
        criteria_texts: List[str],
    ) -> List[float]:
        """Batch compute consistency scores efficiently."""
        # Prepare all inputs
        batch_inputs = []
        cache_keys = []
        uncached_indices = []
        scores = [None] * len(ref_images)
        
        for idx, (ref_img, img, criteria) in enumerate(zip(ref_images, images, criteria_texts)):
            if self.prompt_template == 1:
                hash_key = (hash_pil_image(ref_img), hash_pil_image(img), criteria)
                text_prompt = f"Do images meet the following criteria? {criteria} Please answer Yes or No."
            else:
                hash_key = (hash_pil_image(ref_img), hash_pil_image(img), "")
                text_prompt = "Do the two images maintain consistency in terms of style, logic and identity? Answer \"Yes\" and \"No\" only."
            
            if hash_key in self.cache:
                scores[idx] = self.cache[hash_key]
            else:
                cache_keys.append(hash_key)
                uncached_indices.append(idx)
                batch_inputs.append(self.prepare_input([ref_img, img], text_prompt))
        
        # Batch inference for uncached items
        if batch_inputs:
            outputs = self.llm.generate(
                batch_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=False
            )
            
            for idx, output, cache_key in zip(uncached_indices, outputs, cache_keys):
                score = self._extract_yes_probability(output)
                scores[idx] = score
                self.add_to_cache(cache_key, score)
        
        return scores