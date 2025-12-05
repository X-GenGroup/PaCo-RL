from datasets import Dataset, load_dataset
import pyarrow.parquet as pq
import random
import os
from tqdm import tqdm

def split_dataset(dataset, test_num=128, seed=42, language=None):
    if isinstance(language, str):
        language = [language]

    dataset = [
        {'prompt': item['instruction'], 'image': item['input_image'], 'instruction_language': item['instruction_language']}
        for item in tqdm(dataset, desc="Filtering dataset")
        if language is None or item['instruction_language'] in language
    ]
    dataset = Dataset.from_list(dataset)

    random.seed(seed)
    test_indices = random.sample(range(len(dataset)), test_num)
    train_indices = [i for i in range(len(dataset)) if i not in test_indices]
    test_dataset = [dataset[i] for i in test_indices]
    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = Dataset.from_list(test_dataset)
    train_dataset = Dataset.from_list(train_dataset)
    return train_dataset, test_dataset

def save_train_all(dataset, output_dir='train_all'):
    os.makedirs(output_dir, exist_ok=True)
    print("Full dataset size:", len(dataset))
    table = dataset.data.table
    pq.write_table(table, os.path.join(output_dir, 'train.parquet'))
    pq.write_table(table, os.path.join(output_dir, 'test.parquet'))

def save_train_split(dataset, output_dir='train_split'):
    train_dataset, test_dataset = split_dataset(dataset, test_num=128, seed=42)
    os.makedirs(output_dir, exist_ok=True)
    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))
    pq.write_table(train_dataset.data.table, os.path.join(output_dir, 'train.parquet'))
    pq.write_table(test_dataset.data.table, os.path.join(output_dir, 'test.parquet'))

def save_train_split_cn(dataset, output_dir='train_split_cn'):
    train_dataset, test_dataset = split_dataset(dataset, test_num=128, seed=42, language='cn')
    os.makedirs(output_dir, exist_ok=True)
    print("Train dataset size (CN):", len(train_dataset))
    print("Test dataset size (CN):", len(test_dataset))
    pq.write_table(train_dataset.data.table, os.path.join(output_dir, 'train.parquet'))
    pq.write_table(test_dataset.data.table, os.path.join(output_dir, 'test.parquet'))

def save_train_split_en(dataset, output_dir='train_split_en'):
    train_dataset, test_dataset = split_dataset(dataset, test_num=128, seed=42, language='en')
    os.makedirs(output_dir, exist_ok=True)
    print("Train dataset size (EN):", len(train_dataset))
    print("Test dataset size (EN):", len(test_dataset))
    pq.write_table(train_dataset.data.table, os.path.join(output_dir, 'train.parquet'))
    pq.write_table(test_dataset.data.table, os.path.join(output_dir, 'test.parquet'))

def main():
    dataset = load_dataset('stepfun-ai/GEdit-Bench', split='train')
    output_root = 'dataset/GEdit-Bench'
    # save_train_all(dataset, output_dir='train_all')
    save_train_split(dataset, output_dir=os.path.join(output_root, 'train_split'))
    # save_train_split_cn(dataset, output_dir='train_split_cn')
    # save_train_split_en(dataset, output_dir=os.path.join(output_root, 'train_split_en'))


if __name__ == "__main__":
    main()