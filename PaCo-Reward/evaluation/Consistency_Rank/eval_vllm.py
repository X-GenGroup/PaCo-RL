import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
from scipy.stats import kendalltau, spearmanr
from consistency_scorer_vllm import ConsistencyScorerVLLM
import argparse


def footrule_distance(ref, pred):
    n = len(ref)
    ref_pos = {v: i for i, v in enumerate(ref)}
    pred_pos = {v: i for i, v in enumerate(pred)}
    return sum(abs(ref_pos[i] - pred_pos[i]) for i in ref)


def compute_ranking_metrics(pred_rank, gt_rank):
    metrics = {}
    correct = sum(p == g for p, g in zip(pred_rank, gt_rank))
    metrics['accuracy'] = correct / len(gt_rank)

    kendalltau_corr, _ = kendalltau(pred_rank, gt_rank)
    spearmanr_corr, _ = spearmanr(pred_rank, gt_rank)
    metrics['kendalltau'] = kendalltau_corr
    metrics['spearmanr'] = spearmanr_corr

    footrule_dist = footrule_distance(gt_rank, pred_rank)
    metrics['footrule_distance'] = footrule_dist

    top1_correct = int(pred_rank[0] == gt_rank[0])
    bottom1_correct = int(pred_rank[-1] == gt_rank[-1])
    metrics['top1_accuracy'] = top1_correct
    metrics['bottom1_accuracy'] = bottom1_correct
    metrics['top1_bottom1_accuracy'] = (top1_correct + bottom1_correct) / 2
    return metrics


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Consistency Scorer on Ranking Task (VLLM)")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory containing the dataset')
    parser.add_argument('--model', type=str, default="X-GenGroup/PaCo-Reward-7B", 
                       help='Model path or name')
    parser.add_argument('--prompt_template', type=int, default=3, choices=[1,2,3], 
                       help='''Prompt template version (1, 2 or 3)
                       1. Instruction with criteria
                       2. Instruction for binary answer
                       3. Instruction for binary answer with reasoning''')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, 
                       help='Number of GPUs for tensor parallelism')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, 
                       help='GPU memory utilization ratio')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size for inference')
    args = parser.parse_args()
    return args


def process_item(item, scorer, IMAGE_DIR, PROMPT_DATA, result_file):
    """Process single item synchronously."""
    folder = item['folder']
    candidate_filenames = item['comp_images']
    ref_filename = item['ref_image']
    idx = item['folder']
    
    ref_image_path = os.path.join(IMAGE_DIR, folder, ref_filename)
    ref_image = Image.open(ref_image_path).convert('RGB')
    candidate_image_paths = [os.path.join(IMAGE_DIR, folder, c) for c in candidate_filenames]
    candidate_images = [Image.open(c).convert('RGB') for c in candidate_image_paths]
    ref_images = [ref_image] * len(candidate_image_paths)
    prompt_data = [PROMPT_DATA[idx]] * len(candidate_image_paths)

    # Compute scores
    scores = scorer(ref_images, candidate_images, prompt_data)
    scores = [float(s) for s in scores]
    
    # Compute ranking
    pred_rank = np.argsort(scores)[::-1].tolist()
    ranking = [candidate_filenames[i] for i in pred_rank]
    gt_rank = [candidate_filenames.index(img) for img in item['rank_images']]
    metric_dict = compute_ranking_metrics(pred_rank, gt_rank)

    res_item = {
        'folder': idx,
        "ref_image": ref_filename,
        "comp_images": candidate_filenames,
        'rank_images': item['rank_images'],
        'pred_rank': ranking,
        "scores": scores,
        "rank": pred_rank,
    }
    res_item.update(metric_dict)

    # Write to file
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(res_item) + '\n')
    
    return res_item


def main():
    args = parse_arguments()
    
    # Initialize VLLM-based scorer
    consistency_scorer = ConsistencyScorerVLLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        prompt_template=args.prompt_template,
    )

    # Get the base name
    model_short_name = args.model.split('/')[-1].replace('.', '_')
    result_file = f'./results/{model_short_name}-v{args.prompt_template}.jsonl'

    dataset_dir = args.dataset_dir

    IMAGE_DIR = os.path.join(dataset_dir, "images")
    PROMPT_DATA_FILE = os.path.join(dataset_dir, "prompt_data.jsonl")

    with open(PROMPT_DATA_FILE, 'r', encoding='utf-8') as f:
        PROMPT_DATA = [json.loads(line) for line in f if line.strip()]
        PROMPT_DATA = {item['idx']: item for item in PROMPT_DATA}

    def hash_item(item):
        ref_img = item.get('ref_img') or item.get('ref_image')
        cmp_img = item.get('comparison_images') or item.get('comp_images')
        return (item['folder'], ref_img, tuple(cmp_img))

    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            existing_results = [json.loads(line) for line in f if line.strip()]
        existing_hashes = set(hash_item(item) for item in existing_results)
    else:
        existing_hashes = set()

    print(f"Found {len(existing_hashes)} existing results. Resuming evaluation...")
    
    full_rank_datapath = os.path.join(dataset_dir, "consistency_full_rank.jsonl")
    with open(full_rank_datapath, 'r', encoding='utf-8') as f:
        full_rank_data = [json.loads(line) for line in f if line.strip()]

    # Filter processed items
    full_rank_data = [
        item for item in full_rank_data
        if hash_item(item) not in existing_hashes
    ]

    print(f"To be evaluated: {len(full_rank_data)} items")

    # Process items with progress bar
    for item in tqdm(full_rank_data, desc="Evaluating"):
        process_item(item, consistency_scorer, IMAGE_DIR, PROMPT_DATA, result_file)


if __name__ == "__main__":
    main()