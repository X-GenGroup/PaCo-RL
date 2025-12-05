import os
import json
from openai import AsyncOpenAI
from PIL import Image
import numpy as np
from tqdm.asyncio import tqdm as async_tqdm
import asyncio
from scipy.stats import kendalltau, spearmanr
import argparse


def generate_comparison_pairs(IMAGE_DIR, folders):
    pairs = []
    for cnt, idx in enumerate(folders):
        folder_path = os.path.join(IMAGE_DIR, idx)
        images = sorted([img for img in os.listdir(folder_path) if img.endswith('.png')])
        first_indices, second_indices = zip(*[
            list(map(int, img.split('.')[0].split('_')))
            for img in images
        ])
        first_indices = sorted(set(first_indices))
        second_indices = sorted(set(second_indices))
        
        for i in first_indices:
            for j in second_indices:
                ref_img = f"{i}_{j}.png"
                candidates = [
                    [f"{x}_{y}.png" for x in first_indices]
                    for y in second_indices if y != j
                ]
                for candidate_group in candidates:
                    pairs.append((os.path.join(folder_path, ref_img), 
                                [os.path.join(folder_path, c) for c in candidate_group]))
    return pairs


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
    parser = argparse.ArgumentParser(description="Evaluate Consistency Scorer on Ranking Task")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory containing the dataset')
    parser.add_argument('--model', type=str, default="ConsistencyReward-7B", 
                       help='Model name for consistency scorer')
    parser.add_argument('--prompt_template', type=int, default=1, choices=[1,2,3],
                       help='Prompt template version to use (1, 2, or 3)')
    parser.add_argument('--port', type=int, default=8000, 
                       help='Port number for the model server')
    parser.add_argument('--batch_size', type=int, default=64, 
                       help='Number of items to process in parallel')
    args = parser.parse_args()
    return args


async def process_item(item, scorer, IMAGE_DIR, PROMPT_DATA, result_file, lock):
    """
    Process a single item: compute scores, ranking, and write results.
    """
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
    scores = await scorer(ref_images, candidate_images, prompt_data)
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

    # Append result to file with lock
    async with lock:
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(res_item) + '\n')
    
    return res_item


async def main():
    args = parse_arguments()
    model = args.model
    prompt_template = args.prompt_template
    port = args.port
    batch_size = args.batch_size

    if 'clip' in model.lower():
        from clipI_scorer import ClipIScorer
        scorer = ClipIScorer(device='cuda')
    elif 'dreamsim' in model.lower():
        from dreamsim_scorer import DreamSimScorer
        scorer = DreamSimScorer(device='cuda')
    else:
        from consistency_scorer import ConsistencyScorer
        scorer = ConsistencyScorer(
            model=model,
            base_url=f"http://127.0.0.1:{port}/v1",
            prompt_template=prompt_template,
            max_concurrent=300,
        )

    model_short_name = args.model.split('/')[-1].replace('.', '_')
    result_file = f'./results/{model_short_name}-v{prompt_template}.jsonl'

    dataset_dir = args.dataset_dir
    IMAGE_DIR = os.path.join(dataset_dir, "images")
    PROMPT_DATA_FILE = os.path.join(dataset_dir, "prompt_data.jsonl")

    with open(PROMPT_DATA_FILE, 'r', encoding='utf-8') as f:
        PROMPT_DATA = [json.loads(line) for line in f if line.strip()]
        PROMPT_DATA = {item['idx']: item for item in PROMPT_DATA}

    def hash_item(item):
        if 'ref_img' in item:
            ref_img = item['ref_img']
        elif 'ref_image' in item:
            ref_img = item['ref_image']
        else:
            raise ValueError('No ref image key found')

        if 'comparison_images' in item:
            cmp_img = item['comparison_images']
        elif 'comp_images' in item:
            cmp_img = item['comp_images']
        else:
            raise ValueError('No comparison images key found')
        
        return (item['folder'], ref_img, tuple(cmp_img))

    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            existing_results = [json.loads(line) for line in f if line.strip()]
        existing_hashes = set(hash_item(item) for item in existing_results)
    else:
        existing_hashes = set()

    print("Found", len(existing_hashes), "existing results. Resuming evaluation...")
    full_rank_datapath = os.path.join(dataset_dir, "consistency_full_rank.jsonl")

    with open(full_rank_datapath, 'r', encoding='utf-8') as f:
        full_rank_data = [json.loads(line) for line in f if line.strip()]

    # Filter out already evaluated items
    full_rank_data = [
        item for item in full_rank_data
        if hash_item(item) not in existing_hashes
    ]

    print("To be evaluated items:", len(full_rank_data))

    # Create a lock for file writing
    lock = asyncio.Lock()

    # Batch parallel processing
    tasks = []
    for item in full_rank_data:
        task = process_item(item, scorer, IMAGE_DIR, PROMPT_DATA, result_file, lock)
        tasks.append(task)
        
        # When the number of tasks reaches batch_size, execute a batch
        if len(tasks) >= batch_size:
            await async_tqdm.gather(*tasks, desc="Evaluating")
            tasks = []
    
    # Process remaining tasks
    if tasks:
        await async_tqdm.gather(*tasks, desc="Evaluating")


if __name__ == "__main__":
    asyncio.run(main())