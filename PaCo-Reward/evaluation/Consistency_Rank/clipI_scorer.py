from typing import List, Tuple, Optional, Union
import re
import open_clip
import torch
import numpy as np
from PIL import Image

class ClipIScorer(torch.nn.Module):
    """
        Scorer - measure similarity image pairs
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-H-14",
            pretrained="laion2b_s32b_b79k",
            device=device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")

    @torch.no_grad()
    def __call__(self,
        ref_images: Union[Image.Image, list[Image.Image]],
        images: List[Image.Image],
        metadata : List[dict]
    ) -> np.ndarray:
        if isinstance(ref_images, Image.Image) and isinstance(images, Image.Image):
            ref_images = [ref_images]
            images = [images]
        elif isinstance(images, Image.Image):
            images = [images] * len(ref_images)
        elif isinstance(ref_images, Image.Image):
            ref_images = [ref_images] * len(images)

        scores = np.zeros(len(images), dtype=np.float32)
        for idx, (ref_img, img, meta) in enumerate(zip(ref_images, images, metadata)):

            clip_matrix = self.compute_ClipI_matrix(ref_img, img) # It should be a 1x1 matrix
            scores[idx] = clip_matrix[0,0].item() # Get the only element in the matrix
        
        return scores

    @torch.no_grad()
    def compute_ClipI_matrix(self, ref_images : Union[Image.Image, list[Image.Image]], image : Union[Image.Image, list[Image.Image]]):
        ref_images = [ref_images] if isinstance(ref_images, Image.Image) else ref_images
        input_images = [image] if isinstance(image, Image.Image) else image

        ref_images = torch.stack([self.preprocess(img).to(self.device) for img in ref_images], dim=0)
        input_images = torch.stack([self.preprocess(img).to(self.device) for img in input_images], dim=0)

        with torch.no_grad():
            ref_image_features = self.model.encode_image(ref_images)
            input_image_features = self.model.encode_image(input_images)
            ref_image_features /= ref_image_features.norm(dim=-1, keepdim=True)
            input_image_features /= input_image_features.norm(dim=-1, keepdim=True)

            clip_scores = (input_image_features @ ref_image_features.T)
            return clip_scores.cpu()

def download_model():
    scorer = ClipIScorer(device='cpu')

if __name__ == "__main__":
    download_model()