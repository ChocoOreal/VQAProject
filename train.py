import torch
import torchmetrics.classification as metrics
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    get_scheduler,
    logging,
)
from model import *
from utils import *
import tqdm.auto as tqdm
import dataloader
from datasets import set_caching_enabled

# SET CONSTANT
DATA_PATH="./dataset"


class VQATrainer:
    def __init__(self, model, optimizer, device, train_dataloader, eval_dataloader, num_labels,path_to_checkpoint='./model.pt', num_epochs=5,loss_fnc=nn.CrossEntropyLoss()):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.savepath = path_to_checkpoint
        self.num_epochs = num_epochs
        self.loss_fnc = loss_fnc
        self.num_training_steps = num_epochs * len(train_dataloader)
        self.lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, 
                             num_training_steps=self.num_training_steps)
        self.metric = metrics.MulticlassAccuracy(num_labels).to(device)
    def __call__(self):
        torch.cuda.empty_cache()
        progress_bar = tqdm(range(self.num_training_steps))
        start_epoch, self.model, self.optimizer, loss = load_checkpoint(self.model, self.optimizer, self.device, self.savepath)
        best_score = 0
        start_epoch = 0
        progress_bar.update(start_epoch)
        print (self.optimizer)
        print (self.loss_fnc)
        for epoch in range(start_epoch, self.num_epochs):
            self.model.train()
            running_loss = 0.0
            last_loss = 0.0
            avg_loss = 0.0
            for (i, batch) in enumerate(self.train_dataloader):
                questions = batch['input_ids'].to(self.device )
                imgs = batch['pixel_values'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(questions, imgs, attention_mask, token_type_ids)
                loss = self.loss_fnc(logits, labels)
                loss.backward()
                self.optimizer.step()
#                 if (i % 10 == 9):
#                     for grads in (list(self.model.parameters())[1:3]):
#                         print(grads.grad)
                
                self.lr_scheduler.step()
                
                progress_bar.update(1)
                progress_bar.set_description("Current epoch: {}".format(epoch), refresh=True)
                running_loss += loss.item()
                if (i % 10 == 9):
                    
                    last_loss = running_loss / 10
                    print("batch {} loss: {}".format(i + 1, last_loss))
                    avg_loss += running_loss
                    running_loss = 0
            avg_loss /= (i + 1)
            self.model.eval()
            score  = 0.0
            with torch.no_grad():
                
                for i, vdata in enumerate(self.eval_dataloader):
                    vquests = vdata['input_ids'].to(self.device)
                    vimgs = vdata['pixel_values'].to(self.device)
                    vtoken_type = vdata['token_type_ids'].to(self.device)
                    vattention = vdata['attention_mask'].to(self.device)
                    vlabels = vdata['labels'].to(self.device)
                    voutputs = self.model(vquests, vimgs, vattention, vtoken_type)
                    vpredicts = torch.argmax(voutputs, dim=1)
                    score += self.metric(vpredicts, vlabels)
            avg_score = score / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_score))
            avg_loss = 0.0
            if avg_score > best_score:
                best_score = avg_score
                model_path = self.savepath
                save_checkpoint(
                    epoch,
                    self.model.state_dict(),
                    self.optimizer.state_dict(),
                    loss,
                    model_path
                )


def train(device):
    train_dataloader, test_dataloader, eval_dataloader, answer_space = dataloader.get_data_loader(DATA_PATH)
    model = VQAModel(len(answer_space)).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    trainer = VQATrainer(model, optimizer, device, train_dataloader, eval_dataloader, len(answer_space))
    print ("Model is trained on {}".format(device))
    trainer()

