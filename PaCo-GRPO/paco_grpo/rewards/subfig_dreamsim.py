import os
import torch
from PIL import Image
import numpy as np
from paco_grpo.utils import divide_image, divide_prompt, extract_grid_info
from dreamsim import dreamsim


class SubfigDreamSimScorer():
    """
        Scorer for sub-images DreamSim score - measure similarity between sub-image pairs.
    """
    def __init__(self, device):
        self.device = device
        # Use huggingface cache directory to store the model
        cache_dir = os.environ.get('HF_HOME', './')
        cache_dir = os.path.join(cache_dir, 'dreamsim_models')
        self.model, self.preprocess = dreamsim(pretrained=True, device=device, cache_dir=cache_dir)

    @torch.no_grad()
    def __call__(self,
        images: list[Image.Image],
        prompts : list[str],
        metadata : list[dict]
    ) -> torch.Tensor:
        """
            Compute the average DreamSim score for each subfigure-subprompt pair in a batch of images and prompts.

            Args:
                images (list[Image.Image]): List of PIL images to be evaluated.
                prompts (list[str]): List of prompts corresponding to each image. Each prompt may contain subprompts for subfigures.
                metadata (list[dict]): List of metadata dictionaries for each image (not used in this function).

            Returns:
                torch.Tensor: A tensor containing the average DreamSim scores for each subfigure-subprompt pair.
        """
        scores = []
        for image, prompt in zip(images, prompts):
            grid_info = extract_grid_info(prompt)
            sub_images = divide_image(image, grid_info)
            sub_prompts = divide_prompt(prompt)[1:]

            dreamsim_matrix = self.compute_dreamsim_matrix(sub_images, sub_images).cpu()
            # Select non-diagonal elements
            diag_mask = torch.eye(len(sub_images), dtype=torch.bool)
            # Average non-diagonal elements for each sub-image
            dreamsim_scores = dreamsim_matrix.masked_select(~diag_mask).view(len(sub_images), -1).mean(dim=1).numpy()
            # Average the scores
            scores.append(np.mean(dreamsim_scores))

        return np.array(scores)
    
    @torch.no_grad()
    def compute_dreamsim_matrix(self, ref_images : list[Image.Image], images : list[Image.Image]) -> torch.Tensor:
        ref_images_tensor = torch.cat([self.preprocess(img).to(self.device) for img in ref_images], dim=0)
        input_images_tensor = torch.cat([self.preprocess(img).to(self.device) for img in images], dim=0)

        with torch.no_grad():
            # Compute pairwise distance matrix - dreamsim matrix is symmetric
            dist = torch.stack([
                self.model(ref_images_tensor, input_images_tensor[i:i+1])
                for i in range(len(images))
            ], dim=0)
            sim = 1 - dist  # Convert distance to similarity
        
        return sim.cpu()

def download_model():
    scorer = SubfigDreamSimScorer(device='cpu')

def main():
    image_paths = [
        '/root/siton-data-51d3ce9aba3246f88f64ea65f79d5133/images/base/0/0001_0003.png',
        '/root/siton-data-51d3ce9aba3246f88f64ea65f79d5133/images/consistency_all/180/0001_0003.png'
    ]
    images = [Image.open(img) for img in image_paths]
    prompt = "THREE-PANEL Images with a 1x3 grid layout a male child with a round face, short ginger hair, and curious, wide eyes, rendered in watercolor style.All illustrations maintain a warm, whimsical watercolor aesthetic with soft edges and vibrant yet gentle colors. The child's features, including ginger hair and wide-eyed curiosity, remain consistent across settings. [LEFT]:The child plays in a sunlit backyard, surrounded by scattered toys and a half-built sandcastle. Dandelion puffs float in the air, and a small dog bounds joyfully nearby. The scene emphasizes playful energy with loose brushstrokes and warm golden-green hues. [MIDDLE]:The child explores a museum exhibit, gazing up at a towering dinosaur skeleton. Display cases glow softly with amber lighting, casting playful shadows. His posture leans forward in wonder, clutching a magnifying glass, with watercolor textures suggesting aged parchment and fossil textures. [RIGHT]:The child sits cross-legged in a wooden treehouse, sketching in a notebook. Sunlight filters through leaves, dappling the pages. A jar of fireflies and binoculars rest beside him, with distant hills rendered in hazy blue layers to evoke depth and quiet imagination."
    prompts = [prompt for _ in range(len(images))]

    scorer = SubfigDreamSimScorer(device='cuda:0')

    scores = scorer(images, prompts, [])

    print(scores)

if __name__ == "__main__":
    main()