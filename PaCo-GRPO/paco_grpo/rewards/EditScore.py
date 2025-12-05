# modified from EditScore - https://github.com/VectorSpaceLab/EditScore
import os
from PIL import Image
import numpy as np
import json
import regex as re
import random
import math
import asyncio
from typing import List, Union, Optional
from openai import OpenAI, AsyncOpenAI
from itertools import combinations
import logging

from ..utils import pil_image_to_base64, divide_image, divide_prompt, extract_grid_info

# VLLM log filter
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# -----------------------------------------------------utils------------------------------------------------
def fix_json(input_str):
    # Add double quotes around keys using regex
    fixed_str = re.sub(r'(\w+):', r'"\1":', input_str)
    
    # Add double quotes around string values if necessary and wrap int/float values in []
    def format_value(match):
        key, value, comma = match.groups()
        value = value.strip()
        # Check if value is an integer or float
        if re.match(r'^-?\d+(\.\d+)?$', value):
            value = f'[{value}]'
        # Check if value is a boolean or null
        elif re.match(r'^(true|false|null)$', value, re.IGNORECASE):
            pass  # leave as is
        else:
            # Add quotes around string values
            value = f'"{value}"'
        return f'{key}: {value}{comma}'
    
    fixed_str = re.sub(r'(".*?"):(.*?)(,|})', format_value, fixed_str)
    
    return fixed_str

def repair_reasoning_field_robust(json_str: str) -> str:
    """
    使用正则表达式和先行断言,健壮地修复 "reasoning" 字段内部未转义的双引号。
    此方法可以处理 "reasoning" 字段不是最后一个字段的情况。

    Args:
        json_str: 可能包含格式错误的JSON字符串。

    Returns:
        修复后的JSON字符串。
    """
    # 1. 定义新的正则表达式,使用正向先行断言来定位 "reasoning" 值的结束位置
    # re.DOTALL 标志让 '.' 可以匹配包括换行符在内的任意字符
    pattern = re.compile(
        # --- 第1个捕获组: reasoning 字段的 "前缀" ---
        r'("reasoning"\s*:\s*")'
        
        # --- 第2个捕获组: reasoning 字段的 "内容" ---
        r'(.*?)'
        
        # --- 正向先行断言: 寻找值的结束边界,但不消耗它 ---
        # 匹配到 "reasoning" 值的结束双引号,这个双引号后面必须跟着一个逗号或一个右花括号
        r'(?="\s*[,}])',
        
        re.DOTALL
    )

    # 2. 定义一个更简单的替换函数
    def replacer(match):
        # 提取出两个捕获组
        prefix = match.group(1)      # 例如: '"reasoning" : "'
        content = match.group(2)     # 例如: 'Overall building...'
        
        # 只在 "内容" 部分进行替换,将所有双引号转义
        fixed_content = content.replace('"', '\\"')
        
        # 重新组合。注意:我们不需要处理后缀,因为它没有被匹配和消耗掉。
        return prefix + fixed_content

    # 3. 使用 re.sub 执行查找和替换
    repaired_str = pattern.sub(replacer, json_str)
    
    return repaired_str

def verify(s, target_sequence):
    # Count the occurrences of the target sequence
    count = s.count(target_sequence)

    # Check if the target sequence appears exactly twice
    return count == 2


def is_int_between_0_and_10(s):
    try:
        num = int(s)
        return 0 <= num <= 10
    except ValueError:
        return False
    
def is_str_a_list_of_ints_0_to_10(s):
    try:
        # Attempt to parse the string as a Python literal (list, dict, etc.)
        parsed = ast.literal_eval(s)

        # Check if the parsed object is a list
        if not isinstance(parsed, list):
            return False

        # Check if all elements are integers and between 0 to 10
        return all(isinstance(item, int) and 0 <= item <= 10 for item in parsed)

    except (ValueError, SyntaxError):
        # If parsing fails or any other error occurs
        return False
    
def is_str_valid_score_format_brackets(s):
    try:
        # Removing brackets and splitting the string by commas
        content = s.strip("[]").split(',')

        length = len(content)

        # Parsing each element and checking the format and range
        scores = {}
        for item in content:
            key, value = item.split(':')
            key = key.strip()
            value = int(value.strip())

            # Check if the key starts with 'score' and the value is in the correct range
            if not key.startswith("score") or not 0 <= value <= 10:
                return False

            scores[key] = value

        fetch_words = [f"score{i+1}" for i in range(length)]
        # Check if at least 'score1' and 'score2' are present
        return all(key in scores for key in fetch_words)

    except (ValueError, SyntaxError):
        # If any parsing error occurs
        return False

def normalize_quotes(s: str) -> str:
    """
    Replace curly/smart quotes with normal ASCII quotes.
    """
    # 常见的几种智能引号 U+201C U+201D U+2018 U+2019
    return s.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")

def mllm_output_to_dict(input_string, give_up_parsing=False, text_prompt=None):
    """
    Args:
        input_string (str): actually the output of the mllm model to be parsed
        output_file_name (str): The name of the output file.
    """
    # Catch for gpt4v rate_limit_exceeded error
    if input_string == "rate_limit_exceeded":
        return "rate_limit_exceeded"

    # Define the delimiters
    delimiter = '||V^=^V||'

    if input_string.count(delimiter) == 2:
        if not verify(input_string, delimiter):
            print("The required delimiters were not found correctly in the string.", flush=True)
            return False
        # Extract the content between the delimiters
        start_index = input_string.find(delimiter) + len(delimiter)
        end_index = input_string.rfind(delimiter)
    else:
        # find the json mannually
        # some mllm tends not to output the delimiters, but it does output the json contents
        # so we will find the json content mannually
        start_index = input_string.find('{')
        end_index = input_string.rfind('}') + 1
        if start_index == -1 or end_index == 0:
            # json not found
            # some mllm tends to output only a list of scores like [6, 0], 
            # this time we will just get the scores and ignore the reasoning (other part of the json)
            start_index = input_string.find('[')
            end_index = input_string.rfind(']') + 1
            if give_up_parsing: # if we want to give up parsing
                guessed_value = random.randint(0, 10)
                print(f"1111 Failed to find the json content in the string. Guess a value : {text_prompt=} {input_string=} {guessed_value=}.", flush=True)
                json_content = {'score': [guessed_value], "reasoning": f"guess_if_cannot_parse | {input_string}"}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            elif re.match(r'^\[\d+, ?\d+\]$', input_string[start_index:end_index]):
                scores = json.loads(input_string[start_index:end_index])
                if not isinstance(scores, list):
                    scores = [scores]
                json_content = {'score': scores, "reasoning": "System: output is simply a list of scores"}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            elif is_int_between_0_and_10(input_string): # if output is simply a number
                scores = [int(input_string)]
                json_content = {'score': scores, "reasoning": "System: output is simply a number"}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            else:
                print(f"22 222 Failed to find the json content in the string. {text_prompt=} {input_string=}", flush=True)
                return False
    
    # Check if we found two delimiters
    if start_index != -1 and end_index != -1 and start_index != end_index:
        # Extract the JSON string
        json_str = input_string[start_index:end_index].strip()
        json_str = json_str.replace("\n", "")
        # Parse the JSON string into a dictionary
        try:
            json_str = normalize_quotes(json_str)
            new_data = json.loads(json_str)
            if not isinstance(new_data['score'], list):
                new_data['score'] = [new_data['score']]
        except Exception as e1:
            print(f"Now fixing: {e1=} {json_str=}")
            try:
                new_data = json.loads(fix_json(json_str))
                return new_data
            except Exception as e2:
                try:
                    print(f"Now fixing: {e2=} {fix_json(json_str)=}")
                    new_data = json.loads(repair_reasoning_field_robust(fix_json(json_str)))
                    return new_data
                except Exception as e3:
                    print(f"Error: Cannot fix {e3=} {repair_reasoning_field_robust(fix_json(json_str))=}")
                    return False
        return new_data
    else:
        print("The required delimiters were not found correctly in the string.")
        return False
    
# -------------------------------------------------prompt----------------------------------------------------------------
# This file is generated automatically through parse_prompt.py
_context_no_delimit_reasoning_first = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

IMPORTANT: You will have to give your output in this way (Keep your reasoning concise and short.):
{
"reasoning" : "...",
"score" : [...]
}
"""

_prompts_0shot_two_image_edit_rule = """RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit.
"""

_prompts_0shot_tie_rule_SC = """
From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

Editing instruction: <instruction>
"""

_prompts_0shot_rule_PQ = """RULES:

The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to 10: 
A score from 0 to 10 will be given based on image naturalness. 
(
    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
    10 indicates that the image looks natural.
)
A second score from 0 to 10 will rate the image artifacts. 
(
    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
    10 indicates the image has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]
"""


# -----------------------------------------------------scorer------------------------------------------------

class EditScorer:
    def __init__(
            self,
            client : AsyncOpenAI,
            model: str = "EditScore-7B",
            score_range: int = 25,
    ):
        self.client = client
        self.model = model
        self.score_range = score_range
        self.context = _context_no_delimit_reasoning_first
        self.SC_prompt = "\n".join([self.context, _prompts_0shot_two_image_edit_rule, _prompts_0shot_tie_rule_SC.replace('10', str(self.score_range))])
        self.PQ_prompt = "\n".join([self.context, _prompts_0shot_rule_PQ.replace('10', str(self.score_range))])

    def prepare_SC_prompt(self, instruction):
        return self.SC_prompt.replace('<instruction>', instruction)
    
    def prepare_SC_messages(self, images : List[Image.Image], prompt):
        prompt = self.prepare_SC_prompt(prompt)
        messages = [
            {
                'role': 'user',
                'content': [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": pil_image_to_base64(img)}} for img in images]
            }
        ]
        return messages
    
    def prepare_PQ_messages(self, image : Image.Image):
        messages = [
            {
                'role': 'user',
                'content': [{"type": "text", "text": self.PQ_prompt}, {"type": "image_url", "image_url": {"url": pil_image_to_base64(image)}}]
            }
        ]
        return messages


    async def inference(self, messages):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=512, # args following edit score
            temperature=0.7,
            top_p=0.9,
        )
        return response.choices[0].message.content
    
    def __call__(self, images: List[Image.Image], prompts: List[str], metadata : List[dict]) -> np.ndarray:
        assert len(images) == len(prompts) == len(metadata), "Length of images, prompts and metadata must be the same."
        def check_metadata_keys(prompt, meta):
            necessary_keys = ['ref_image']
            for key in necessary_keys:
                if key not in meta:
                    raise ValueError(f"Metadata missing necessary key: {key} in {meta} for {prompt}")

        for meta, prompt in zip(metadata, prompts):
            check_metadata_keys(prompt, meta)

        scores = asyncio.run(self._score_all_async(images, prompts, metadata))
        return np.array(scores)
    
    async def _score_all_async(self, images: List[Image.Image], prompts: List[str], metadata: List[dict]):
        """异步批量评分所有项目"""
        PQ_tasks = []
        SC_tasks = []
        for image, prompt, meta in zip(images, prompts, metadata):
            PQ_tasks.append(self.score_PQ_async(image, prompt, meta))
            ref_image = meta.get('ref_image')
            SC_tasks.append(self.score_SC_async([image, ref_image], prompt, meta))
        
        # 并行执行所有 PQ 评分
        PQ_scores = await asyncio.gather(*PQ_tasks)
        # 并行执行所有 SC 评分
        SC_scores = await asyncio.gather(*SC_tasks)
        # 计算最终分数
        scores = []
        for PQ_score, SC_score in zip(PQ_scores, SC_scores):
            overall_score = SC_score # only use SC score
            overall_score = math.sqrt(SC_score * PQ_score) # Follow EditScore to use geometric mean
            scores.append(overall_score)

        return scores

    async def score_PQ_async(self, image: Image.Image, prompt : str, metadata : dict):
        PQ_messages = self.prepare_PQ_messages(image)
        output = await self.inference(PQ_messages)
        result = mllm_output_to_dict(output, give_up_parsing=False, text_prompt=prompt)
        # If it is well parsed, normalize the score to [0, 1]
        if result and result != "rate_limit_exceeded" and 'score' in result and isinstance(result['score'], list) and len(result['score']) == 2:
            score = result['score']
        else:
            score = [0.0, 0.0]

        return np.mean(score) / self.score_range
    
    async def score_SC_async(self, images: List[Image.Image], prompt : str, metadata : dict):
        SC_messages = self.prepare_SC_messages(images, prompt)
        output = await self.inference(SC_messages)
        result = mllm_output_to_dict(output, give_up_parsing=False, text_prompt=prompt)
        # If it is well parsed, normalize the score to [0, 1]
        if result and result != "rate_limit_exceeded" and 'score' in result and isinstance(result['score'], list) and len(result['score']) == 2:
            score = result['score']
        else:
            score = [0.0, 0.0]

        return np.mean(score) / self.score_range