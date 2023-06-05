import torch
import os

SAVE_PATH = "./model.pt"
def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, loss, save_path=SAVE_PATH):
    torch.save({
      "epoch": epoch,
      "model_state_dict": model_state_dict,
      "optimizer_state_dict": optimizer_state_dict,
      "loss": loss
    }, save_path)

def load_checkpoint(model, optimizer, device, path=SAVE_PATH):
    epoch = 0
    loss = None
    if (os.path.isfile(path)):
        print(f"=> Loading checkpoint from {path}")
        loaded = torch.load(path, map_location=device)
        epoch = loaded["epoch"]
        model.load_state_dict(loaded["model_state_dict"])
        optimizer.load_state_dict(loaded["optimizer_state_dict"])
        loss = loaded["loss"]
    else:
        print ("Cant find filename {}".format(path))
    return epoch, model, optimizer, loss