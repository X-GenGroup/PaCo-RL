# editscore/__init__.py
import sys
sys.path.insert(0, 'editscore')

from typing import Optional
from .utils import (
    mllm_output_to_dict
)
import math
from . import vie_prompts
import numpy as np
from .json_parser import parse_vlm_output_to_dict

from typing import Optional, List, Dict
import numpy as np
import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

class EditScore:
    def __init__(
        self,
        backbone="gpt-4.1",
        openai_url="https://api.openai.com/v1/chat/completions",
        key=None,
        model_name_or_path="",
        score_range: int=25,
        temperature: float=0.7,
        tensor_parallel_size: int=1,
        gpu_memory_utilization: float=0.9,
        max_model_len: int=1536,
        max_num_batched_tokens: int=1536,
        max_num_seqs: int=32,
        num_pass: int=1,
        reduction: str="average_last",
        seed: int=42,
        lora_path: Optional[str]=None,
        cache_dir: Optional[str]=None,
    ) -> None:
        self.backbone = backbone
        self.score_range = score_range
        self.reduction = reduction
        self.seed = seed
        self.num_pass = num_pass

        if self.backbone == 'openai':
            from .mllm_tools.openai import GPT4o
            self.model = GPT4o(key, model_name=model_name_or_path, url=openai_url)
        elif self.backbone == "qwen25vl":
            from .mllm_tools.qwen25vl import Qwen25VL
            self.model = Qwen25VL(
                vlm_model=model_name_or_path,
                temperature=temperature,
                seed=seed,
                lora_path=lora_path,
            )
        elif self.backbone == "qwen25vl_vllm":
            from .mllm_tools.qwen25vl_vllm import Qwen25VL
            self.model = Qwen25VL(
                vlm_model=model_name_or_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                temperature=temperature,
                seed=seed,
                lora_path=lora_path,
                cache_dir=cache_dir,
            )
        elif self.backbone == "internvl3_5":
            from .mllm_tools.internvl35_lmdeploy import InternVL35
            self.model = InternVL35(model=model_name_or_path, tensor_parallel_size=tensor_parallel_size)
            
        self.context = vie_prompts._context_no_delimit_reasoning_first
    
        self.SC_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_two_image_edit_rule, vie_prompts._prompts_0shot_tie_rule_SC.replace('10', str(self.score_range))])
        self.PQ_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_rule_PQ.replace('10', str(self.score_range))])

    def evaluate(self, image_prompts, text_prompt):
        if not isinstance(image_prompts, list):
            image_prompts = [image_prompts]

        if self.backbone in ['openai']:
            self.model.use_encode = False if isinstance(image_prompts[0], str) else True
            
        _SC_prompt = self.SC_prompt.replace("<instruction>", text_prompt)
        
        SC_prompt_final = self.model.prepare_input(image_prompts, _SC_prompt)
        PQ_prompt_final = self.model.prepare_input(image_prompts[-1], self.PQ_prompt) # assume the last image is the edited image

        outputs_multi_pass = []

        for i in range(self.num_pass):
            SC_dict = False
            PQ_dict = False
            tries = 0
            max_tries = 2
            while SC_dict is False or PQ_dict is False:
                tries += 1
                give_up_parsing = True if tries > max_tries else False

                result_SC = self.model.inference(SC_prompt_final, seed=self.seed + i)
                result_PQ = self.model.inference(PQ_prompt_final, seed=self.seed + i)

                if result_SC in ["I'm sorry, but I can't assist with that request."] or result_PQ in ["I'm sorry, but I can't assist with that request."]:
                    give_up_parsing = True
                    
                SC_dict = mllm_output_to_dict(result_SC, give_up_parsing=give_up_parsing, text_prompt=text_prompt, score_range=self.score_range)
                PQ_dict = mllm_output_to_dict(result_PQ, give_up_parsing=give_up_parsing, text_prompt=text_prompt, score_range=self.score_range)

            if SC_dict == "rate_limit_exceeded" or PQ_dict == "rate_limit_exceeded":
                print("rate_limit_exceeded") 
                raise ValueError("rate_limit_exceeded")
            
            try:
                SC_score = min(SC_dict['score']) / (self.score_range / 10)
                PQ_score = min(PQ_dict['score']) / (self.score_range / 10)
                O_score = math.sqrt(SC_score * PQ_score)
            except Exception as e:
                print(f"{e=} {SC_dict['score']=} {PQ_dict['score']=}")
                raise e

            try:
                outputs_multi_pass.append({
                    'prompt_following': SC_dict['score'][0] / (self.score_range / 10),
                    'consistency': SC_dict['score'][1] / (self.score_range / 10),
                    'perceptual_quality': PQ_score,
                    'overall': O_score,
                })
            except Exception as e:
                print(f"{e=} {SC_dict['score']=} {PQ_dict['score']=}")
                raise e
        
        output = {
                    "prompt_following": np.mean([output_per_pass["prompt_following"] for output_per_pass in outputs_multi_pass]),
                    "consistency": np.mean([output_per_pass["consistency"] for output_per_pass in outputs_multi_pass]),
                    "perceptual_quality": np.mean([output_per_pass["perceptual_quality"] for output_per_pass in outputs_multi_pass]),
                    "overall": np.mean([output_per_pass["overall"] for output_per_pass in outputs_multi_pass]),
                    "SC_reasoning": SC_dict["reasoning"],
                    "PQ_reasoning": PQ_dict["reasoning"],
                }
        if self.reduction == "average_first":
            output["overall"] = math.sqrt(output["prompt_following"] * output["perceptual_quality"])
        return output


    def batch_evaluate(self, image_prompts, text_prompt):
        SC_prompt = [self.SC_prompt.replace("<instruction>", _text_prompt) for _text_prompt in text_prompt]

        SC_prompt = [self.model.prepare_input(image_prompt, _SC_prompt) for image_prompt, _SC_prompt in zip(image_prompts, SC_prompt)]
        PQ_prompt = [self.model.prepare_input(image_prompt, self.PQ_prompt) for image_prompt in image_prompts]

        outputs_multi_pass = [[] for _ in range(len(image_prompts))]
        for i in range(self.num_pass):
            results = self.model.batch_inference(SC_prompt + PQ_prompt, seed=self.seed + i)

            SC_evaluations = [parse_vlm_output_to_dict(results[i]) for i in range(len(results) // 2)]
            PQ_evaluations = [parse_vlm_output_to_dict(results[i]) for i in range(len(results) // 2, len(results))]

            for idx, (SC_evaluation, PQ_evaluation) in enumerate(zip(SC_evaluations, PQ_evaluations)):
                SC_scores = SC_evaluation["score"]
                PQ_scores = PQ_evaluation["score"]

                if len(SC_scores) == 0:
                    SC_scores = [self.score_range / 2]
                if len(PQ_scores) == 0:
                    PQ_scores = [self.score_range / 2]

                SC_score = min(SC_scores) / (self.score_range / 10)
                PQ_score = min(PQ_scores) / (self.score_range / 10)
                if SC_score < 0 or SC_score > 10:
                    SC_score = self.score_range / 2
                if PQ_score < 0 or PQ_score > 10:
                    PQ_score = self.score_range / 2
                O_score = math.sqrt(SC_score * PQ_score)

                outputs_multi_pass[idx].append(
                    {
                        "SC_score": SC_score,
                        "PQ_score": PQ_score,
                        "O_score": O_score,
                        "SC_score_reasoning": SC_evaluation["reasoning"],
                        "PQ_score_reasoning": PQ_evaluation["reasoning"],
                        "SC_raw_output": results[idx],
                        "PQ_raw_output": results[len(results) // 2 + idx],
                    }
                )
        
        outputs = []
        for idx, outputs_per_prompt in enumerate(outputs_multi_pass):
            outputs.append(
                {
                    "SC_score": np.mean([output_per_pass["SC_score"] for output_per_pass in outputs_per_prompt]),
                    "PQ_score": np.mean([output_per_pass["PQ_score"] for output_per_pass in outputs_per_prompt]),
                    "O_score": np.mean([output_per_pass["O_score"] for output_per_pass in outputs_per_prompt]),
                    "SC_score_reasoning": outputs_per_prompt[0]["SC_score_reasoning"],
                    "PQ_score_reasoning": outputs_per_prompt[0]["PQ_score_reasoning"],
                    "SC_raw_output": outputs_per_prompt[0]["SC_raw_output"],
                    "PQ_raw_output": outputs_per_prompt[0]["PQ_raw_output"],
                }
            )
            if self.reduction == "average_first":
                outputs[-1]["O_score"] = math.sqrt(outputs[-1]["SC_score"] * outputs[-1]["PQ_score"])
        return outputs

class ConsistencyScore:
    def __init__(
        self,
        backbone="qwen25vl_vllm",
        model_name_or_path="",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 1,
        max_model_len: int = 4096,
        max_num_batched_tokens: int = 8192,
        max_num_seqs: int = 32,
        num_pass: int = 1,
        seed: int = 42,
        lora_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize ConsistencyScore for calculating yes/no token probabilities.
        
        Args:
            backbone: Model backbone type (currently supports qwen25vl_vllm)
            model_name_or_path: Path to the model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum model sequence length
            max_num_batched_tokens: Maximum number of batched tokens
            max_num_seqs: Maximum number of sequences
            num_pass: Number of passes for evaluation (for averaging)
            seed: Random seed for reproducibility
            lora_path: Path to LoRA weights if using fine-tuned model
            cache_dir: Cache directory for merged LoRA models
        """
        self.backbone = backbone
        self.seed = seed
        self.num_pass = num_pass
        self.temperature = temperature

        if self.backbone == "qwen25vl_vllm":
            from editscore.mllm_tools.qwen25vl_vllm import Qwen25VL
            # Initialize the model without temperature since we need logprobs
            self.model = Qwen25VL(
                vlm_model=model_name_or_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                temperature=temperature,
                seed=seed,
                lora_path=lora_path,
                cache_dir=cache_dir,
            )
            # Store processor for tokenization
            self.processor = AutoProcessor.from_pretrained(
                cache_dir if cache_dir and lora_path else model_name_or_path
            )
        else:
            raise NotImplementedError(f"Backbone {backbone} not supported for ConsistencyScore")
    
    def _get_yes_no_tokens(self) -> Dict[str, List[int]]:
        """
        Get all token IDs for 'yes' and 'no' variants (case-insensitive).
        
        Returns:
            Dictionary with 'yes' and 'no' keys containing lists of token IDs
        """
        tokenizer = self.processor.tokenizer
        
        # Get all variants of yes/no tokens
        yes_variants = ['yes', 'Yes', 'YES', 'y', 'Y']
        no_variants = ['no', 'No', 'NO', 'n', 'N']
        
        yes_tokens = []
        no_tokens = []
        
        # Tokenize each variant and collect unique token IDs
        for variant in yes_variants:
            tokens = tokenizer.encode(variant, add_special_tokens=False)
            # Only consider single-token representations
            if len(tokens) == 1:
                yes_tokens.extend(tokens)
        
        for variant in no_variants:
            tokens = tokenizer.encode(variant, add_special_tokens=False)
            # Only consider single-token representations
            if len(tokens) == 1:
                no_tokens.extend(tokens)
        
        # Remove duplicates
        yes_tokens = list(set(yes_tokens))
        no_tokens = list(set(no_tokens))
        
        return {'yes': yes_tokens, 'no': no_tokens}
    
    def _compute_yes_no_probs(self, logprobs_dict, yes_tokens, no_tokens):
        """
        计算 yes 和 no tokens 的概率总和
        
        Args:
            logprobs_dict: 第一个 token 的 logprobs 字典
            yes_tokens: yes token IDs 列表
            no_tokens: no token IDs 列表
            
        Returns:
            (yes_prob, no_prob) 元组
        """
        yes_prob = sum(np.exp(logprobs_dict[tid].logprob) 
                      for tid in yes_tokens if tid in logprobs_dict)
        no_prob = sum(np.exp(logprobs_dict[tid].logprob) 
                     for tid in no_tokens if tid in logprobs_dict)
        return yes_prob, no_prob
    
    def _generate_with_logprobs(self, messages, pass_idx):
        """
        生成带有 logprobs 的输出
        
        Args:
            messages: 输入消息
            pass_idx: 当前 pass 的索引
            
        Returns:
            模型输出
        """
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=1,
            logprobs=20,
            seed=self.seed + pass_idx
        )
        return self.model.model.generate(messages, sampling_params, use_tqdm=False)
    
    def evaluate(self, image_prompts, text_prompt: str) -> Dict[str, float]:
        """
        Evaluate the consistency score based on yes/no token probabilities.
        
        Args:
            image_prompts: Input images (can be single image or list)
            text_prompt: Text prompt/question to evaluate
            
        Returns:
            Dictionary containing scores
        """
        if not isinstance(image_prompts, list):
            image_prompts = [image_prompts]
        
        # Get yes/no token IDs
        token_ids = self._get_yes_no_tokens()
        yes_tokens = token_ids['yes']
        no_tokens = token_ids['no']

        # Prepare prompts
        consistency_text_prompt = f"Given two images after an edit with instruction '{text_prompt}', do they maintain consistency in terms of style, logic and identity? Answer 'Yes' or 'No' only."
        prompt_following_text_prompt = f"Given two images after an edit with instruction '{text_prompt}', is the edit following the instruction well? Answer 'Yes' or 'No' only."
        
        # Prepare inputs
        consistency_messages = self.model.prepare_input(image_prompts, consistency_text_prompt)
        prompt_following_messages = self.model.prepare_input(image_prompts, prompt_following_text_prompt)
        
        # Collect probabilities across multiple passes
        consistency_yes_probs = []
        consistency_no_probs = []
        prompt_following_yes_probs = []
        prompt_following_no_probs = []

        for i in range(self.num_pass):
            # Generate for consistency
            consistency_outputs = self._generate_with_logprobs(consistency_messages, i)
            if consistency_outputs[0].outputs[0].logprobs:
                first_token_logprobs = consistency_outputs[0].outputs[0].logprobs[0]
                yes_prob, no_prob = self._compute_yes_no_probs(first_token_logprobs, yes_tokens, no_tokens)
                consistency_yes_probs.append(yes_prob)
                consistency_no_probs.append(no_prob)

            # Generate for prompt following
            prompt_following_outputs = self._generate_with_logprobs(prompt_following_messages, i)
            if prompt_following_outputs[0].outputs[0].logprobs:
                first_token_logprobs = prompt_following_outputs[0].outputs[0].logprobs[0]
                yes_prob, no_prob = self._compute_yes_no_probs(first_token_logprobs, yes_tokens, no_tokens)
                prompt_following_yes_probs.append(yes_prob)
                prompt_following_no_probs.append(no_prob)

        # Calculate conditional probabilities
        def calc_conditional_prob(yes_probs, no_probs):
            avg_yes = np.mean(yes_probs) if yes_probs else 0.0
            avg_no = np.mean(no_probs) if no_probs else 0.0
            total = avg_yes + avg_no
            return avg_yes / total if total > 0 else 0.0
        
        consistency_score = calc_conditional_prob(consistency_yes_probs, consistency_no_probs)
        prompt_following_score = calc_conditional_prob(prompt_following_yes_probs, prompt_following_no_probs)

        return {
            "prompt_following": prompt_following_score,
            "consistency": consistency_score,
            "perceptual_quality": 0,
            "overall": math.sqrt(prompt_following_score * consistency_score),
            "SC_reasoning": 'none',
            "PQ_reasoning": 'none'
        }
    
    def batch_evaluate(self, image_prompts_list: List, text_prompts: List[str]) -> List[Dict[str, float]]:
        """
        Batch evaluation for multiple image-text pairs.
        
        Args:
            image_prompts_list: List of image prompts (each can be single image or list)
            text_prompts: List of text prompts corresponding to images
            
        Returns:
            List of dictionaries containing scores for each input pair
        """
        # Get yes/no token IDs
        token_ids = self._get_yes_no_tokens()
        yes_tokens = token_ids['yes']
        no_tokens = token_ids['no']
        
        # Prepare all inputs for both consistency and prompt following
        consistency_messages_list = []
        prompt_following_messages_list = []
        
        for image_prompts, text_prompt in zip(image_prompts_list, text_prompts):
            if not isinstance(image_prompts, list):
                image_prompts = [image_prompts]
            
            consistency_text_prompt = f"Given two images after an edit with instruction '{text_prompt}', do they maintain consistency in terms of style, logic and identity? Answer 'Yes' or 'No' only."
            prompt_following_text_prompt = f"Given two images after an edit with instruction '{text_prompt}', is the edit following the instruction well? Answer 'Yes' or 'No' only."
            
            consistency_messages_list.append(self.model.prepare_input(image_prompts, consistency_text_prompt))
            prompt_following_messages_list.append(self.model.prepare_input(image_prompts, prompt_following_text_prompt))
        
        # Initialize results storage
        n_samples = len(image_prompts_list)
        results = [{
            'consistency_yes_probs': [],
            'consistency_no_probs': [],
            'prompt_following_yes_probs': [],
            'prompt_following_no_probs': []
        } for _ in range(n_samples)]
        
        # Collect probabilities across multiple passes
        for i in range(self.num_pass):
            sampling_params = SamplingParams(
                max_tokens=1,
                temperature=1,
                logprobs=20,
                seed=self.seed + i
            )
            
            # Batch generate for consistency
            consistency_outputs = self.model.model.generate(
                consistency_messages_list, sampling_params, use_tqdm=False
            )
            
            # Batch generate for prompt following
            prompt_following_outputs = self.model.model.generate(
                prompt_following_messages_list, sampling_params, use_tqdm=False
            )
            
            # Process consistency outputs
            for idx, output in enumerate(consistency_outputs):
                if output.outputs[0].logprobs:
                    first_token_logprobs = output.outputs[0].logprobs[0]
                    yes_prob, no_prob = self._compute_yes_no_probs(
                        first_token_logprobs, yes_tokens, no_tokens
                    )
                    results[idx]['consistency_yes_probs'].append(yes_prob)
                    results[idx]['consistency_no_probs'].append(no_prob)
            
            # Process prompt following outputs
            for idx, output in enumerate(prompt_following_outputs):
                if output.outputs[0].logprobs:
                    first_token_logprobs = output.outputs[0].logprobs[0]
                    yes_prob, no_prob = self._compute_yes_no_probs(
                        first_token_logprobs, yes_tokens, no_tokens
                    )
                    results[idx]['prompt_following_yes_probs'].append(yes_prob)
                    results[idx]['prompt_following_no_probs'].append(no_prob)
        
        # Compute final scores for each input
        final_results = []
        for result in results:
            # Calculate conditional probabilities
            def calc_conditional_prob(yes_probs, no_probs):
                avg_yes = np.mean(yes_probs) if yes_probs else 0.0
                avg_no = np.mean(no_probs) if no_probs else 0.0
                total = avg_yes + avg_no
                return avg_yes / total if total > 0 else 0.0
            
            consistency_score = calc_conditional_prob(
                result['consistency_yes_probs'], 
                result['consistency_no_probs']
            )
            prompt_following_score = calc_conditional_prob(
                result['prompt_following_yes_probs'], 
                result['prompt_following_no_probs']
            )
            
            final_results.append({
                "prompt_following": prompt_following_score,
                "consistency": consistency_score,
                "perceptual_quality": 0,
                "overall": math.sqrt(prompt_following_score * consistency_score),
                "SC_reasoning": 'none',
                "PQ_reasoning": 'none'
            })
        
        return final_results