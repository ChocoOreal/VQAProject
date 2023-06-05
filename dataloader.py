from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
import torch
import os
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
)
class VQADataset(Dataset):
    def __init__(self, dataset: Dataset, answer_space, tokenizer, feature_extractor, transform=None):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.images = [os.path.join(".", "dataset", "images", img +".png") for img in dataset['image_id']]
        self.questions = dataset['question']
        self.answers = dataset.map(lambda example: {
            "labels": answer_space.index(example['answer'].replace(" ", "").split(",")[0])
        })
        
    def __len__(self):
        return len(self.questions)
    def __getitem__(self, idx):
        tokens = self.tokenizer(self.questions[idx], padding="max_length", truncation=True, max_length=100)
        if (isinstance(idx, slice)):
            PILImg = []
            for path_to_img in self.images[idx]:
                PILImg.append(Image.open(path_to_img).convert("RGB"))
        else:
            PILImg = Image.open(self.images[idx]).convert("RGB")

        pixel_values = self.feature_extractor(PILImg, return_tensors="pt")
        pixel_values = pixel_values["pixel_values"]
        if (self.transform):
            ans = self.transform(torch.as_tensor(self.answers[idx]['labels']))
            input_ids = self.transform(torch.as_tensor(tokens['input_ids']))
            token_type_ids = self.transform(torch.as_tensor(tokens['token_type_ids']))
            attention_mask = self.transform(torch.as_tensor(tokens['attention_mask']))
        else:
            ans = torch.as_tensor(self.answers[idx]['labels'])
            input_ids = torch.as_tensor(tokens['input_ids'])
            token_type_ids = torch.as_tensor(tokens['token_type_ids'])
            attention_mask = torch.as_tensor(tokens['attention_mask'])
        return {
            "input_ids": input_ids.squeeze(),
            "token_type_ids": token_type_ids.squeeze(),
            "attention_mask": attention_mask.squeeze(),
            "pixel_values": pixel_values.squeeze(),
            "labels": ans.squeeze()
        }
    
def get_data_loader_and_answer(path_to_dataset):
    dataset = load_dataset("csv", 
                       data_files=path_to_dataset
                    )
    after_split = dataset['train'].train_test_split(test_size=0.2)
    dataset['train'] = after_split['train']
    dataset['test'] = after_split['test']
    split_train = dataset['train'].train_test_split(test_size=0.2)
    dataset['train'] = split_train['train']
    dataset['eval'] = split_train['test']

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    with open(os.path.join(path_to_dataset, "answer_space.txt"), "r") as f:
        answer_space = f.read().splitlines()

    train_set = VQADataset(dataset['train'], answer_space, tokenizer, image_processor)
    test_set = VQADataset(dataset['test'], answer_space, tokenizer, image_processor)
    eval_set = VQADataset(dataset['eval'], answer_space, tokenizer, image_processor)

    train_dataloader = DataLoader(train_set, batch_size=32, num_workers=2, pin_memory=False)
    test_dataloader = DataLoader(test_set, batch_size=32, num_workers=2, pin_memory=False)
    eval_dataloader =  DataLoader(eval_set, batch_size=32, num_workers=2, pin_memory=False)

    return train_dataloader, test_dataloader, eval_dataloader, answer_space