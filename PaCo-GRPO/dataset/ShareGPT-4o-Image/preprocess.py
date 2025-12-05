import argparse
import os
import random
import json
# Get absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

random.seed(42)

def preprocess_data(dataset_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    datafile_path = os.path.join(dataset_dir, 'text_and_image_to_image.json')
    with open(datafile_path, 'r', encoding='utf-8') as f:
        data = json.load(f)


    new_data = []
    for item in data:
        for input_image in item['input_image']:
            new_item = {
                'ref_image': input_image,
                'prompt': item['input_prompt'],
                'metadata': item
            }
            new_data.append(new_item)

    test_num = 128
    random.shuffle(new_data)
    test_data = new_data[:test_num]
    train_data = new_data[test_num:]
    print("Total number of data:", len(new_data))
    print("Total number of train data:", len(train_data), " ({:.2f}%)".format(len(train_data)/len(new_data)*100))
    print("Total number of test data:", len(test_data), " ({:.2f}%)".format(len(test_data)/len(new_data)*100))
    with open(os.path.join(output_dir, 'train_metadata.jsonl'), 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    with open(os.path.join(output_dir, 'test_metadata.jsonl'), 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess ShareGPT-4o-Image Dataset")
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the raw dataset directory')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    preprocess_data(args.dataset_dir, args.dataset_dir)