from typing import List, Tuple, Optional, Union
import re
import open_clip
import torch
import numpy as np
import math
from PIL import Image
from paco_grpo.utils import divide_image, extract_grid_info, divide_prompt, hash_pil_image

class SubfigClipTScorer(torch.nn.Module):
    """
        Scorer for sub-images clip-T-score - align Image-Text semantics.
    """
    def __init__(self, device, max_cache_size: int = 1024):
        super().__init__()
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-H-14",
            pretrained="laion2b_s32b_b79k",
            device=device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")
        self.max_cache_size = max_cache_size if max_cache_size is not None else math.inf
        self.cache : dict[Tuple[str, str], float] = {} # (img_hash, prompt) -> score


    def add_to_cache(self, key: Tuple[str, str], value: float):
        if len(self.cache) >= self.max_cache_size:
            # Remove the oldest item in the cache (FIFO)
            self.cache.pop(next(iter(self.cache)))

        self.cache[key] = value

    @torch.no_grad()
    def __call__(self,
        images: List[Image.Image],
        prompts : List[str],
        metadata : List[dict]
    ) -> np.ndarray:
        """
            Compute the average CLIP score for each subfigure-subprompt pair in a batch of images and prompts.

            Args:
                images (List[Image.Image]): List of PIL images to be evaluated.
                prompts (List[str]): List of prompts corresponding to each image. Each prompt may contain subprompts for subfigures.
                metadata (List[dict]): List of metadata dictionaries for each image (not used in this function).

            Returns:
                np.ndarray: Array of average CLIP scores for each image, computed as the mean of the diagonal of the subfigure-subprompt CLIP score matrix.
        """
        scores = np.empty(len(images), dtype=np.float64)
        
        for batch_idx, (image, prompt) in enumerate(zip(images, prompts)):
            grid_info = extract_grid_info(prompt)
            sub_images = divide_image(image, grid_info)
            sub_prompts = divide_prompt(prompt)[1:]

            clip_scores = np.empty(len(sub_images), dtype=np.float64)
            uncached_pairs = []
            uncached_indices = []
            
            for i, (sub_img, sub_txt) in enumerate(zip(sub_images, sub_prompts)):
                img_hash = hash_pil_image(sub_img)
                key = (img_hash, sub_txt)
                
                if key in self.cache:
                    clip_scores[i] = self.cache[key]
                else:
                    uncached_pairs.append((sub_img, sub_txt, key))
                    uncached_indices.append(i)

            if uncached_pairs:
                uncached_images = [pair[0] for pair in uncached_pairs]
                uncached_texts = [pair[1] for pair in uncached_pairs]

                text_tokens = self.tokenizer(uncached_texts).to(self.device)
                images_tensor = torch.stack([
                    self.preprocess(img).to(self.device)
                    for img in uncached_images
                ], dim=0)
                
                image_features = self.model.encode_image(images_tensor)
                text_features = self.model.encode_text(text_tokens)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                # A faster way to compute the diagonal of the matrix - no need to compute the full matrix
                subfig_clip_scores = (image_features * text_features).sum(dim=-1).cpu().numpy()

                # Update cache and clip_scores
                for idx, score, (_, _, key) in zip(uncached_indices, subfig_clip_scores, uncached_pairs):
                    clip_scores[idx] = float(score)
                    self.add_to_cache(key, float(score))

            scores[batch_idx] = np.mean(clip_scores)
        
        return scores

    @torch.no_grad()
    def compute_ClipT_matrix(self, text : Union[str, List[str]], image : Union[Image.Image, list[Image.Image]]):
        input_texts = [text] if isinstance(text, str) else text
        input_images = [image] if isinstance(image, Image.Image) else image

        text_tokens = self.tokenizer(input_texts).to(self.device)
        images = torch.stack([self.preprocess(img).to(self.device) for img in input_images], dim=0)

        with torch.no_grad():
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            clip_scores = (image_features @ text_features.T)
            return clip_scores.cpu()

def download_model():
    scorer = SubfigClipTScorer(device='cpu')        

def main():
    image_paths = ['/root/flux_trained_0001_0003.png', '/root/flux_0001_0003.png', '/root/sd3_0001_0003.png']
    images = [Image.open(img) for img in image_paths]
    prompt = "THREE-PANEL Images with a 1x3 grid layout a male child with a round face, short ginger hair, and curious, wide eyes, rendered in watercolor style.All illustrations maintain a warm, whimsical watercolor aesthetic with soft edges and vibrant yet gentle colors. The child's features, including ginger hair and wide-eyed curiosity, remain consistent across settings. [LEFT]:The child plays in a sunlit backyard, surrounded by scattered toys and a half-built sandcastle. Dandelion puffs float in the air, and a small dog bounds joyfully nearby. The scene emphasizes playful energy with loose brushstrokes and warm golden-green hues. [MIDDLE]:The child explores a museum exhibit, gazing up at a towering dinosaur skeleton. Display cases glow softly with amber lighting, casting playful shadows. His posture leans forward in wonder, clutching a magnifying glass, with watercolor textures suggesting aged parchment and fossil textures. [RIGHT]:The child sits cross-legged in a wooden treehouse, sketching in a notebook. Sunlight filters through leaves, dappling the pages. A jar of fireflies and binoculars rest beside him, with distant hills rendered in hazy blue layers to evoke depth and quiet imagination."
    prompts = [prompt for _ in range(len(images))]

    scorer = SubfigClipTScorer(device='cuda:0')

    scores = scorer(images, prompts, [])

    print(scores)

if __name__ == "__main__":
    download_model()
    # main()