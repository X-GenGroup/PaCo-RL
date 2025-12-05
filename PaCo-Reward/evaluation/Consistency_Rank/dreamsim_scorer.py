import os
import torch
from PIL import Image
import numpy as np
from dreamsim import dreamsim


class DreamSimScorer():
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
        ref_images: list[Image.Image],
        images: list[Image.Image],
        metadata : list[dict]
    ) -> torch.Tensor:
        if isinstance(ref_images, Image.Image) and isinstance(images, Image.Image):
            ref_images = [ref_images]
            images = [images]
        elif isinstance(images, Image.Image):
            images = [images] * len(ref_images)
        elif isinstance(ref_images, Image.Image):
            ref_images = [ref_images] * len(images)

        scores = np.zeros(len(images), dtype=np.float32)
        for idx, (ref_img, img, meta) in enumerate(zip(ref_images, images, metadata)):
            dreamsim_matrix = self.compute_dreamsim_matrix([ref_img], [img]).cpu()
            scores[idx] = dreamsim_matrix[0,0].item() # Get the only element in the matrix

        return scores
    
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
    scorer = DreamSimScorer(device='cpu')

if __name__ == "__main__":
    download_model()