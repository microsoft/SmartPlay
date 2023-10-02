'''
Implements the EMMA model
'''

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from numpy import sqrt as sqrt
from transformers import AutoModel, AutoTokenizer

from messenger.models.utils import nonzero_mean, Encoder

class EMMA(nn.Module):
    def __init__(self, state_h=10, state_w=10, action_dim=5, hist_len=3, n_latent_var=128,
                emb_dim=256, f_maps=64, kernel_size=2, n_hidden_layers=1, device=None):
        
        super().__init__()

        # calculate dimensions after flattening the conv layer output
        lin_dim = f_maps * (state_h - (kernel_size - 1)) * (
            state_w - (kernel_size - 1))
        self.conv = nn.Conv2d(hist_len*256, f_maps, kernel_size) # conv layer

        self.state_h = state_h
        self.state_w = state_w
        self.action_dim = action_dim
        self.emb_dim = emb_dim
        self.attn_scale = sqrt(emb_dim)
    
        self.sprite_emb = nn.Embedding(25, emb_dim, padding_idx=0) # sprite embedding layer
        
        hidden_layers = (nn.Linear(n_latent_var, n_latent_var), nn.LeakyReLU())*n_hidden_layers
        self.action_layer = nn.Sequential(
                nn.Linear(lin_dim, n_latent_var),
                nn.LeakyReLU(),
                *hidden_layers,
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic 
        self.value_layer = nn.Sequential(
                nn.Linear(lin_dim, n_latent_var),
                nn.LeakyReLU(),
                *hidden_layers,
                nn.Linear(n_latent_var, 1)
                )

        # key value transforms
        self.txt_key = nn.Linear(768, emb_dim)
        self.scale_key = nn.Sequential(
            nn.Linear(768, 1),
            nn.Softmax(dim=-2)
        )
        
        self.txt_val = nn.Linear(768, emb_dim)
        self.scale_val = nn.Sequential(
            nn.Linear(768, 1),
            nn.Softmax(dim=-2)
        )

        if device:
            self.device = device
        else:
            self.device = torch.device("cpu")

        # get the text encoder
        text_model = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.encoder = Encoder(model=text_model, tokenizer=tokenizer, device=self.device)
        self.to(device)

    def to(self, device):
        '''
        Override the .to() method so that we can store the device as an attribute
        and also update the device for self.encoder (which does not inherit nn.Module)
        '''
        self.device = device
        self.encoder.to(device)
        return super().to(device)

    def attention(self, query, key, value):
        '''
        Cell by cell attention mechanism. Uses the sprite embeddings as query. Key is
        text embeddings
        '''
        kq = query @ key.t() # dot product attention
        mask = (kq != 0) # keep zeroed-out entries zero
        kq = kq / self.attn_scale # scale to prevent vanishing grads
        weights = F.softmax(kq, dim=-1) * mask
        return torch.mean(weights.unsqueeze(-1) * value, dim=-2), weights
        
    def forward(self, obs, manual):
        # encoder the text
        temb = self.encoder.encode(manual)

        # split the observation tensor into objects and avatar
        entity_obs = obs["entities"]
        avatar_obs = obs["avatar"]

        # embedding for the avatar object, which will not attend to text
        avatar_emb = nonzero_mean(self.sprite_emb(avatar_obs))

        # take the non_zero mean of embedded objects, which will act as attention query
        query = nonzero_mean(self.sprite_emb(entity_obs))

        # Attention        
        key = self.txt_key(temb)
        key_scale = self.scale_key(temb) # (num sent, sent_len, 1)
        key = key * key_scale
        key = torch.sum(key, dim=1)
        
        value = self.txt_val(temb)
        val_scale = self.scale_val(temb)
        value = value * val_scale
        value = torch.sum(value, dim=1)
        
        obs_emb, weights = self.attention(query, key, value)

        # compress the channels from KHWC to HWC' where K is history length
        obs_emb = obs_emb.view(self.state_h, self.state_w, -1)
        avatar_emb = avatar_emb.view(self.state_h, self.state_w, -1)
        obs_emb = (obs_emb + avatar_emb) / 2.0

        # permute from HWC to NCHW and do convolution
        obs_emb = obs_emb.permute(2, 0, 1).unsqueeze(0)
        obs_emb = F.leaky_relu(self.conv(obs_emb)).view(-1)
        
        action_probs = self.action_layer(obs_emb)

        action = torch.argmax(action_probs).item()
        if random.random() < 0.05: # random action with 0.05 prob
            action = random.randrange(0, self.action_dim)
        return action