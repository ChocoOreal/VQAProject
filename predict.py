import os
import torch
from model import VQAModel
from torch.optim import AdamW
from utils import *

DATA_PATH = "./dataset"


def deploy_model(device):
    with open(os.path.join(DATA_PATH, "answer_space.txt"), "r") as f:
        answer_space = f.read().splitlines()
    model = VQAModel(len(answer_space)).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    load_checkpoint(model, optimizer, device, './model.pt')
    return model, answer_space




def predict(pixel_values, tokens, model, answer_space):
    model.eval()
    with torch.no_grad():
        output = model(tokens['input_ids'], pixel_values, tokens['attention_mask'], tokens['token_type_ids'])
    prediction = torch.argmax(output, dim=1)
    return answer_space[prediction]