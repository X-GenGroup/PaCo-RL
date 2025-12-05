import os
import io
import json
from torch.utils.data import Dataset
from PIL import Image
import pyarrow.parquet as pq
import pyarrow as pa

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas
    
class GenevalPromptImageDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.dataset = dataset
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        item = {
            "prompt": self.prompts[idx],
            "metadata": self.metadatas[idx]
        }
        # Assuming 'image' in metadata contains a path to the image file
        image_path = self.metadatas[idx]['image']
        image = Image.open(os.path.join(self.dataset, image_path)).convert('RGB')
        item["ref_image"] = image
        return item

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        ref_images = [example["ref_image"] for example in examples]
        return prompts, metadatas, ref_images

class ArrowPromptImageDataset(Dataset):
    def __init__(self, dataset, split='train'):
        arrow_path = os.path.join(dataset, f'{split}.arrow')
        parquet_path = os.path.join(dataset, f'{split}.parquet')
        
        if os.path.exists(arrow_path):
            self.table = pa.ipc.open_file(arrow_path).read_all()
        elif os.path.exists(parquet_path):
            self.table = pq.read_table(parquet_path)
        else:
            raise FileNotFoundError(
                f"No .arrow or .parquet file found for split '{split}' in {dataset}"
            )
    
    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        row = self.table.slice(idx, 1).to_pydict()
        prompt = row['prompt'][0]
        
        image_data = row['image'][0]
        if isinstance(image_data, dict) and 'bytes' in image_data:
            image = Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
        elif isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            raise ValueError(f"Unexpected image format: {type(image_data)}")
        
        return {
            "prompt": prompt,
            "ref_image": image,
            "metadata": {}
        }
    
    @staticmethod
    def collate_fn(examples):
        return (
            [ex["prompt"] for ex in examples],
            [ex["metadata"] for ex in examples],
            [ex["ref_image"] for ex in examples]
        )