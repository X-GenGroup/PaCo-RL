from collections import defaultdict
from typing import List, Tuple, Callable, Union, Dict, Optional
import io
import inspect

from PIL import Image
import numpy as np
import torch
from openai import OpenAI, AsyncOpenAI
from paco_grpo.utils import tensor_list_to_pil_image, tensor_to_pil_image, numpy_to_pil_image, numpy_list_to_pil_image

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):

        buffers = [io.BytesIO() for _ in images]

        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)

        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn

def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta

    return _fn

def clip_score(device):
    from paco_grpo.rewards.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images: List[Image.Image], prompts: List[str], metadata: List[dict]) -> Tuple[np.ndarray, dict]:
        # Convert PIL images to pixel tensors in [0, 1] range
        images = np.stack([np.array(img) / 255.0 for img in images])
        images = torch.tensor(images, dtype=torch.float32).to(device)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def image_similarity_score(device):
    from paco_grpo.rewards.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device).cuda()

    def _fn(images, ref_images):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8)/255.0
        scores = scorer.image_similarity(images, ref_images)
        return scores, {}

    return _fn

def edit_score():
    from paco_grpo.rewards.EditScore import EditScorer

    client = AsyncOpenAI(
        api_key='dummy-key',
        base_url='http://127.0.0.1:8000/v1'
    )

    def _fn(images: List[Image.Image], prompts: List[str], metadatas: List[dict]) -> Tuple[np.ndarray, dict]:
        """
        Compute EditScore for a batch of images and prompts.
        Make sure that each metadata contains 'ref_image' key with the reference image.
        """
        scorer = EditScorer(
            client=client,
            model='EditScore-7B',
            score_range=25
        )
        scores = scorer(images, prompts, metadatas)
        return scores, {}
    
    return _fn

def consistency_for_editing(model='PaCo-Reward-7B', port=8000):
    from paco_grpo.rewards.consistency_for_editing import ConsistencyScorerForEditing

    client = AsyncOpenAI(
        api_key='dummy-key',
        base_url=f'http://127.0.0.1:{port}/v1'
    )

    def _fn(images: List[Image.Image], prompts: List[str], metadatas: List[dict]) -> Tuple[np.ndarray, dict]:
        scorer = ConsistencyScorerForEditing(
            client=client,
            model=model,
        )
        scores = scorer(images, prompts, metadatas)
        return scores, {}

    return _fn

def pickscore_score(device):
    from paco_grpo.rewards.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images : List[Image.Image], prompts : List[str], metadata : List[dict]) -> Tuple[List[float], dict]:
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def imagereward_score(device):
    from paco_grpo.rewards.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):

        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def grid_layout_score():
    import asyncio
    from paco_grpo.rewards.layout_scorer import GridLayoutScorer

    client = AsyncOpenAI(
        api_key='dummy-key',
        base_url='http://127.0.0.1:8000/v1'
    )

    def _fn(images : List[Image.Image], prompts : List[str], metadatas : List[dict]) -> Tuple[List[float], dict]:
        # Create the GridLayoutScorer instance inside the function, to create its own semaphore for this call
        scorer = GridLayoutScorer(
            client=client,
            model='Qwen2.5-VL-7B-Instruct',
            max_concurrent=100, # Adjust based on the system's capabilities (especially when using vllm as local model server)
        )
        scores = scorer(images, prompts, metadatas)
        return scores, {}

    return _fn

def consistency_score(model='Qwen2.5-VL-7B-Instruct', port=8000, prompt_template_version=1):
    import asyncio
    from paco_grpo.rewards.consistency_scorer import ConsistencyScorer

    client = AsyncOpenAI(
        api_key='dummy-key',
        base_url=f'http://127.0.0.1:{port}/v1'
    )

    def _fn(images : List[Image.Image], prompts : List[str], metadatas : List[dict]) -> Tuple[List[float], dict]:
        # Create the ConsistencyScorer instance inside the function, to create its own semaphore for this call
        scorer = ConsistencyScorer(
            client=client,
            model=model,
            max_concurrent=120, # Adjust based on the system's capabilities (especially when using vllm as local model server)
            prompt_template_version=prompt_template_version
        )
        scores = scorer(images, prompts, metadatas)
        return scores, {}

    return _fn

def subfig_clipT_score(device):
    from paco_grpo.rewards.subfig_clipT import SubfigClipTScorer

    scorer = SubfigClipTScorer(device=device)

    def _fn(images : List[Image.Image], prompts : List[str], metadatas : List[dict]) -> Tuple[np.ndarray, dict]:
        scores = scorer(images, prompts, metadatas)
        return scores, {}

    return _fn

def subfig_clipI_score(device):
    from paco_grpo.rewards.subfig_clipI import SubfigClipIScorer

    scorer = SubfigClipIScorer(device=device)

    def _fn(images : List[Image.Image], prompts : List[str], metadatas : List[dict]) -> Tuple[np.ndarray, dict]:
        scores = scorer(images, prompts, metadatas)
        return scores, {}

    return _fn

def subfig_dreamsim_score(device):
    from paco_grpo.rewards.subfig_dreamsim import SubfigDreamSimScorer

    scorer = SubfigDreamSimScorer(device=device)

    def _fn(images : List[Image.Image], prompts : List[str], metadatas : List[dict]) -> Tuple[np.ndarray, dict]:
        scores = scorer(images, prompts, metadatas)
        return scores, {}

    return _fn

def ocr_score(device):
    from paco_grpo.rewards.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images : List[Image.Image], prompts : List[str], metadata : List[dict]):
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def multi_score(
    score_dict: Dict[str, float],
    aggregate_fn: Optional[Callable[[Dict[str, np.ndarray]], np.ndarray]] = None,
    **reward_fn_kwargs,
) -> Callable[[List[Image.Image], List[str], List[dict]], Tuple[dict[str, np.ndarray], dict]]:
    """
    Constructs a multi-score reward function that computes multiple reward metrics for a batch of images and prompts.

    Args:
        device: The device (e.g., "cuda" or "cpu") on which to run the reward functions.
        
        score_dict (List[str]): A dictionary mapping reward function names to their weights.
        
        aggregate_fn (Callable[[Dict[str, np.ndarray]], np.ndarray], optional): A function to aggregate multiple scores.
            If None, defaults to summing all weighted scores along axis 0. The function should accept keyword arguments where:
            - Each keyword corresponds to a key in score_dict (e.g., "clipscore", "aesthetic").
            - Each value is the weighted score array (original_score * weight) for that reward metric.
            - Returns a numpy array representing the final aggregated scores for each sample in the batch.

            Examples:
            - lambda **kwargs: np.sum(list(kwargs.values()), axis=0)  # Sum all weighted scores (default)
            - lambda **kwargs: np.mean(list(kwargs.values()), axis=0)  # Average of weighted scores  
            - lambda clipscore, aesthetic: np.maximum(clipscore, aesthetic)  # Element-wise maximum
            - lambda **kwargs: np.prod(list(kwargs.values()), axis=0)  # Product of all scores

    Returns:
        Callable: A function that takes as input:
            - images (List[Image.Image] or np.ndarray or torch.Tensor): The batch of images to evaluate.
            - prompts (List[str]): The corresponding text prompts for the images.
            - metadata (List[dict]): Additional metadata for each image/prompt pair.
            - ref_images (optional): Reference images for similarity-based rewards.

        The returned function outputs:
            - A dictionary mapping reward names to their computed numpy arrays, including an "avg" key for the aggregated score.
            - A dictionary containing detailed reward information (e.g., per-group or strict scores).

    Raises:
        ValueError: If an unknown score name is provided in score_dict.

    Example:
        reward_fn = multi_score("cuda:0", {"clipscore": 0.5, "aesthetic": 0.5}, aggregate_fn=lambda score1, score2: score1 + score2)
        rewards, details = reward_fn(images, prompts, metadata)
    """
    if aggregate_fn is None:
        # If not given, use np.sum directly
        aggregate_fn = lambda **kwargs: np.sum(list(kwargs.values()), axis=0)

    assert aggregate_fn is not None

    score_functions = {
        "ocr": ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "jpeg_compressibility": jpeg_compressibility,
        "clipscore": clip_score,
        "consistency_score": consistency_score,
        "subfig_clipT": subfig_clipT_score,
        'subfig_clipI': subfig_clipI_score,
        'subfig_dreamsim': subfig_dreamsim_score,
        "grid_layout": grid_layout_score,
        'edit_score': edit_score,
        'consistency_for_editing': consistency_for_editing,
    }

    score_fns = {}

    for score_name, weight in score_dict.items():
        factory = score_functions.get(score_name)
        if factory is None:
            raise ValueError(f"Unknown score: {score_name}")
        params = inspect.signature(factory).parameters
        # Filter reward_fn_kwargs to only include those accepted by the factory
        filtered_kwargs = {k: v for k, v in reward_fn_kwargs.get(score_name, {}).items() if k in params}
        score_fns[score_name] = factory(**filtered_kwargs)

    def _fn(
        images : Union[List[Image.Image], torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray]],
        prompts : List[str],
        metadata: List[dict],
        ref_images: Optional[Union[List[Image.Image], torch.Tensor, np.ndarray]] = None
    ) -> Tuple[dict[str, np.ndarray], dict]:
        # aggregated_scores = {}
        score_details = {}

        # Convert images to PIL format if they are tensors or numpy arrays
        if isinstance(images, torch.Tensor):
            images = tensor_to_pil_image(images)
        elif isinstance(images, np.ndarray):
            images = numpy_to_pil_image(images)
        elif isinstance(images, list) and all(isinstance(img, torch.Tensor) for img in images):
            images = tensor_list_to_pil_image(images)
        elif isinstance(images, list) and all(isinstance(img, np.ndarray) for img in images):
            images = numpy_list_to_pil_image(images)

        assert all(isinstance(img, Image.Image) for img in images), "All images must be a list of PIL Image, or a numpy array / torch Tensor, or a list of them."

        if ref_images is not None:
            # Convert ref_images to PIL format if they are tensors or numpy arrays
            if isinstance(ref_images, torch.Tensor):
                ref_images = tensor_to_pil_image(ref_images)
            elif isinstance(ref_images, np.ndarray):
                ref_images = numpy_to_pil_image(ref_images)
            elif isinstance(ref_images, list) and all(isinstance(img, torch.Tensor) for img in ref_images):
                ref_images = tensor_list_to_pil_image(ref_images)
            elif isinstance(ref_images, list) and all(isinstance(img, np.ndarray) for img in ref_images):
                ref_images = numpy_list_to_pil_image(ref_images)

            assert all(isinstance(img, Image.Image) for img in ref_images), "All ref_images must be a list of PIL Image, or a numpy array / torch Tensor, or a list of them."

            # Add ref_images to metadata for similarity-based/editing-based rewards
            for i in range(len(metadata)):
                metadata[i]['ref_image'] = ref_images[i]

        for score_name, weight in score_dict.items():
            scores, rewards = score_fns[score_name](images, prompts, metadata)

            # Make sure to convert all scores to numpy arrays
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            
            if isinstance(scores, list):
                scores = np.array(scores)

            score_details[score_name] = scores
            # Scale each reward by corresponding weight
            # aggregated_scores[score_name] = weight * scores

        # Aggregate scores from different reward models - do it in stat_tracker instead
        # aggregated_scores = aggregate_fn(**aggregated_scores)

        # score_details['avg'] = aggregated_scores
        return score_details, {}

    return _fn


# ------------------------------------------------Rewards for grid-layout consistency-subfig------------------------------------------------


def main():
    import torchvision.transforms as transforms

    image_paths = [
        "nasa.jpg",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = [{}]  # Example metadata
    score_dict = {
        "clipscore": 1.0
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()