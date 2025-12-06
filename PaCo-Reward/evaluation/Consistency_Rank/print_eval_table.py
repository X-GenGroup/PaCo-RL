import os
import json
import numpy as np
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate and Summarize Ranking Metrics")
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing results from different methods')
    parser.add_argument('--output_file', type=str, default='results_summary.txt',
                        help='Output file for summary results')
    parser.add_argument('--output_format', type=str, default='txt', choices=['txt', 'latex'],
                        help='Output format: txt or latex')
    parser.add_argument('--print_std', action='store_true',
                        help='Whether to print standard deviation alongside mean')
    return parser.parse_args()


def main():
    args = parse_args()
    
    metric_keys = [
        'accuracy', 'kendalltau', 'spearmanr', 'top1_bottom1_accuracy'
    ]
    
    better_direction = defaultdict(lambda: '⬆️')
    better_direction.update({'footrule_distance': '⬇️'})
    
    latex_direction = defaultdict(lambda: '$\\uparrow$')
    latex_direction.update({'footrule_distance': '$\\downarrow$'})
    
    metric_display_names = {
        'accuracy': 'Accuracy',
        'kendalltau': 'Kendall $\\tau$',
        'spearmanr': 'Spearman $\\rho$',
        'top1_bottom1_accuracy': 'T1B1 Acc.',
    }
    
    # Load all results
    all_res = {}
    for method in os.listdir(args.results_dir):
        res_file = os.path.join(args.results_dir, method)
        if not res_file.endswith('.jsonl'):
            continue
            
        with open(res_file, 'r') as f:
            data = [json.loads(line) for line in f]
            
            all_res[method] = {
                key: np.array([item[key] for item in data]) for key in metric_keys
            }
    
    all_res = dict(sorted(all_res.items()))
    
    # Output based on format
    if args.output_format == 'txt':
        write_txt_table(all_res, metric_keys, better_direction, args)
    else:
        write_latex_table(all_res, metric_keys, latex_direction, metric_display_names, args)


def write_txt_table(all_res, metric_keys, better_direction, args):
    with open(args.output_file, 'w') as f_out:
        max_method_len = max(len(m) for m in all_res.keys())
        col_width = 30 if args.print_std else 20
        
        header = f"{'Method':<{max_method_len}} | " + " | ".join(
            f"{key + ' ' + better_direction[key]:^{col_width}}" for key in metric_keys
        )
        separator = "-" * len(header)
        
        print(separator, file=f_out)
        print(header, file=f_out)
        print(separator, file=f_out)
        
        for method, res in all_res.items():
            row = f"{method:<{max_method_len}} | "
            for key in metric_keys:
                mean_val = np.mean(res[key])
                std_val = np.std(res[key])
                cell = f"{mean_val:.4f}±{std_val:.4f}" if args.print_std else f"{mean_val:.4f}"
                row += f"{cell:^{col_width}} | "
            print(row, file=f_out)
        
        print(separator, file=f_out)


def write_latex_table(all_res, metric_keys, latex_direction, metric_display_names, args):
    with open(args.output_file, 'w') as f_out:
        col_nums = 4
        for i in range(0, len(metric_keys), col_nums):
            keys = metric_keys[i:i+col_nums]
            
            col_spec = '|l|' + 'c|' * len(keys)
            latex_table = f"\\begin{{tabular}}{{{col_spec}}}\n\\hline\n"
            latex_table += "Method & " + " & ".join(
                metric_display_names[k] + ' ' + latex_direction[k] for k in keys
            ) + " \\\\\n\\hline\n"
            
            for method, res in all_res.items():
                row = method.replace('_', ' ')
                for key in keys:
                    mean_val = np.mean(res[key])
                    std_val = np.std(res[key])
                    row += f" & {mean_val:.3f} $\\pm$ {std_val:.3f}" if args.print_std else f" & {mean_val:.3f}"
                row += " \\\\\n"
                latex_table += row
            
            latex_table += "\\hline\n\\end{tabular}\n\n"
            print(latex_table, file=f_out)


if __name__ == '__main__':
    main()