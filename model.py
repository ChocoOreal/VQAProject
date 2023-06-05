import torch
import torch.nn as nn
from transformers import (
    ViTModel,
    BertModel
)

class ImageEncoder(nn.Module):
    def __init__(self, dim_out):
        super(ImageEncoder, self).__init__()
        self.image_extractor = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.fc = nn.Sequential(
            nn.Linear(self.image_extractor.config.hidden_size, dim_out),
            nn.Dropout(p=0.5),
            nn.Tanh()
        )

  
    def forward(self, pixel_values):
        # We leave out the first token since it's the CLS token
        features = self.image_extractor(pixel_values=pixel_values).last_hidden_state[:, 1:, :] #(batch_size, sequence_len, hidden_size) 
        output = self.fc(features) # (batch_size, sequence_len, dim_out)
        return output

class Attention(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Attention, self).__init__()
        self.ff_image = nn.Linear(dim_in, dim_out)
        self.ff_question = nn.Linear (dim_in, dim_out)
        self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(dim_out, 1)
        self.tanh = nn.Tanh()
        

    def forward(self, vi, vq): 
        image = self.ff_image(vi) # (batch_size, sequence_len, dim_out)
        ques = self.ff_question(vq).unsqueeze(dim=1) # (batch_size, 1,dim_out)
        ha = self.tanh (image + ques) # (batch_size, sequence_len, dim_out)
        ha = self.dropout(ha)
        att = self.ff_attention(ha) # (batch_size, sequence_len, 1)
        pi = torch.softmax(att, dim=1) # (batch_size, sequence_len, 1)
        vi_attended = (vi * pi).sum(dim=1) # (batch_size, dim_in)
        u = vi_attended + vq 
        return u # (batch_size, dim_in)

class VQAModel(nn.Module):
    def __init__(self, vocab_size, num_attention_layers=2):
        super(VQAModel, self).__init__()
        # We want to take the pooler_output from the model
        self.text_extractor = BertModel.from_pretrained('bert-base-uncased') # (batch_size, hidden_size)
        self.image_extractor = ImageEncoder(self.text_extractor.config.hidden_size)
        self.num_attention_layers = num_attention_layers
        self.sans = nn.ModuleList([Attention(self.text_extractor.config.hidden_size, 512)] * num_attention_layers)
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.text_extractor.config.hidden_size, vocab_size)
        )

    def forward(self, sample_input_ids, sample_pixel_values, sample_attention_mask=None, sample_token_type_ids=None):
        vq = self.text_extractor(input_ids=sample_input_ids,
                                attention_mask=sample_attention_mask,
                                token_type_ids=sample_token_type_ids,
                                ).pooler_output # (batch_size, hidden_size)
        vi = self.image_extractor(sample_pixel_values) # (batch_size, sequence_len, hidden_size)
        for att_layer in self.sans:
            vq = att_layer(vi, vq)
        output = self.mlp(vq) # (batch_size, vocab)
        return output