import os
import glob
import json
import numpy as np

import argparse

PROMPT_FOLLOWING = "prompt_following"
CONSISTENCY = "consistency"
OVERALL = "overall"
SCORE_CATEGORIES = [PROMPT_FOLLOWING, CONSISTENCY, OVERALL]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="qwen25vl_vllm", choices=["qwen25vl", "qwen25vl_vllm", "openai", "internvl3_5"])
    return parser.parse_args()

def main(args):
    result_dir = os.path.join(args.result_dir, args.backbone)
    task_types = sorted(os.listdir(result_dir))

    print(task_types)

    prompt_following_results = dict()
    consistency_results = dict()
    overall_results = dict()

    all_prompt_following_scores = []
    all_consistency_scores = []
    all_overall_scores = []

    for task_type in task_types:
        task_type_dir =  os.path.join(result_dir, task_type)
        prompt_following_json_file = os.path.join(task_type_dir, f"{PROMPT_FOLLOWING}.jsonl")
        consistency_json_file = os.path.join(task_type_dir, f"{CONSISTENCY}.jsonl")
        overall_json_file = os.path.join(task_type_dir, f"{OVERALL}.jsonl")

        total_num = 0
        strict_correct_num = 0
        geq_correct_num = 0
        with open(prompt_following_json_file, 'r') as f:
            for line in f:
                json_line = json.loads(line)
                if json_line['score'][0] > json_line['score'][1]:
                    strict_correct_num += 1
                if json_line['score'][0] >= json_line['score'][1]:
                    geq_correct_num += 1
                total_num += 1
                all_prompt_following_scores.append(json_line['score'][0])
                all_prompt_following_scores.append(json_line['score'][1])
        prompt_following_results[task_type] = [strict_correct_num / total_num, geq_correct_num / total_num]
        
        total_num = 0
        strict_correct_num = 0
        geq_correct_num = 0
        with open(consistency_json_file, 'r') as f:
            for line in f:
                json_line = json.loads(line)
                if json_line['score'][0] > json_line['score'][1]:
                    strict_correct_num += 1
                if json_line['score'][0] >= json_line['score'][1]:
                    geq_correct_num += 1
                total_num += 1
                all_consistency_scores.append(json_line['score'][0])
                all_consistency_scores.append(json_line['score'][1])
        consistency_results[task_type] = [strict_correct_num / total_num, geq_correct_num / total_num]

        total_num = 0
        strict_correct_num = 0
        geq_correct_num = 0
        with open(overall_json_file, 'r') as f:
            for line in f:
                json_line = json.loads(line)
                if json_line['score'][0] > json_line['score'][1]:
                    strict_correct_num += 1
                if json_line['score'][0] >= json_line['score'][1]:
                    geq_correct_num += 1
                total_num += 1
                all_overall_scores.append(json_line['score'][0])
                all_overall_scores.append(json_line['score'][1])
        overall_results[task_type] = [strict_correct_num / total_num, geq_correct_num / total_num]

    prompt_following_results['strict_average'], prompt_following_results['geq_average'] = [
        float(x)
        for x in np.array(list(prompt_following_results.values())).mean(axis=0)
    ]
    consistency_results['strict_average'], consistency_results['geq_average'] = [
        float(x)
        for x in np.array(list(consistency_results.values())).mean(axis=0)
    ]
    overall_results['strict_average'], overall_results['geq_average'] = [
        float(x)
        for x in np.array(list(overall_results.values())).mean(axis=0)
    ]

    print(list(overall_results.keys()))

    task_types = [
        'background_change', 'color_alter', 'style_change', 'subject-add', 'subject-remove', 'subject-replace', 'material_alter',
        'motion_change', 'ps_human', 'text_change', 'tone_transfer', 'extract', 'compose', 'strict_average', 'geq_average'
    ]

    print(" & ".join(task_types))
    print("Prompt Following: " + " & ".join([
        f"{prompt_following_results[task_type]:.3f}" if isinstance(prompt_following_results[task_type], float)
        else f"{prompt_following_results[task_type][0]:.3f}, {prompt_following_results[task_type][1]:.3f}"
        for task_type in task_types
        ])
    )
    print("Consistency: " + " & ".join([
        f"{consistency_results[task_type]:.3f}" if isinstance(consistency_results[task_type], float)
        else f"{consistency_results[task_type][0]:.3f}, {consistency_results[task_type][1]:.3f}"
        for task_type in task_types
    ]))
    print("Overall: " + " & ".join([
        f"{overall_results[task_type]:.3f}" if isinstance(overall_results[task_type], float)
        else f"{overall_results[task_type][0]:.3f}, {overall_results[task_type][1]:.3f}"
        for task_type in task_types
    ]))

    groups = {
        'object': ['subject-add', 'subject-remove', 'subject-replace'],
        'appearance': ['color_alter', 'material_alter', 'style_change', 'tone_transfer'],
        'scene': ['background_change', 'extract'],
        'advanced': ['ps_human', 'text_change', 'motion_change', 'compose'],
    }

    print("--------------------------------")
    print("--------------------------------")

    for group_name, group_task_types in groups.items():
        print(group_name + ":")
        print("Prompt Following & Consistency & Overall")
        prompt_following_mean_strict = np.mean([prompt_following_results[task_type] for task_type in group_task_types], axis=0)[0]
        prompt_following_mean_geq = np.mean([prompt_following_results[task_type] for task_type in group_task_types], axis=0)[1]
        consistency_mean_strict = np.mean([consistency_results[task_type] for task_type in group_task_types], axis=0)[0]
        consistency_mean_geq = np.mean([consistency_results[task_type] for task_type in group_task_types], axis=0)[1]
        overall_mean_strict = np.mean([overall_results[task_type] for task_type in group_task_types], axis=0)[0]
        overall_mean_geq = np.mean([overall_results[task_type] for task_type in group_task_types], axis=0)[1]
        print(f"{prompt_following_mean_strict:.3f} & {consistency_mean_strict:.3f} & {overall_mean_strict:.3f}")
        print(f"{prompt_following_mean_geq:.3f} & {consistency_mean_geq:.3f} & {overall_mean_geq:.3f}")

    print("Average:")
    print("Prompt Following & Consistency & Overall")
    print(f"{prompt_following_results['strict_average']:.3f} & {consistency_results['strict_average']:.3f} & {overall_results['strict_average']:.3f}")
    print(f"{prompt_following_results['geq_average']:.3f} & {consistency_results['geq_average']:.3f} & {overall_results['geq_average']:.3f}")

    print("Prompt Following Scores:")
    print("Min & Max & Mean & Std")
    print(f"{np.min(all_prompt_following_scores):.3f} & {np.max(all_prompt_following_scores):.3f} & {np.mean(all_prompt_following_scores):.3f} & {np.std(all_prompt_following_scores):.3f}")
    print("Consistency Scores:")
    print("Min & Max & Mean & Std")
    print(f"{np.min(all_consistency_scores):.3f} & {np.max(all_consistency_scores):.3f} & {np.mean(all_consistency_scores):.3f} & {np.std(all_consistency_scores):.3f}")
    print("Overall Scores:")
    print("Min & Max & Mean & Std")
    print(f"{np.min(all_overall_scores):.3f} & {np.max(all_overall_scores):.3f} & {np.mean(all_overall_scores):.3f} & {np.std(all_overall_scores):.3f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)