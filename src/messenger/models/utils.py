'''
Common code and utilities used by the models
'''

import torch

class ObservationBuffer:
    '''
    Maintains a buffer of observations along the 0-dim. Observations
    are currently expected to be a dict of np arrays. Currently keeps
    observations in a list and then stacks them via torch.stack().
    TODO: pre-allocate memory for faster calls to get_obs().
    
    Parameters:
    buffer_size
        How many previous observations to track in the buffer
    device
        The device on which buffers are loaded into
    '''
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.buffer = None
        self.device = device

    def _np_to_tensor(self, obs):
        return torch.from_numpy(obs).long().to(self.device)

    def reset(self, obs):
        # initialize / reset the buffer with the observation
        self.buffer = [obs for _ in range(self.buffer_size)]

    def update(self, obs):
        # update the buffer by appending newest observation
        assert self.buffer, "Please initialize buffer first with reset()"
        del self.buffer[0] # delete the oldest entry
        self.buffer.append(obs) # append the newest observation

    def get_obs(self):
        # get a stack of all observations currently in the buffer
        stacked_obs = {}
        for key in self.buffer[0].keys():
            stacked_obs[key] = torch.stack(
                [self._np_to_tensor(obs[key]) for obs in self.buffer]
            )
        return stacked_obs


class Encoder:
    '''
    Text-encoder class with caching for fast sentence-encoding.
    self.encoder and self.tokenizer are expected to be HuggingFace model
    and its respective tokenizer. 
    
    Warning: currently have not implemented max cache size, watch out for
    out of memory errors if the number of possible inputs is v. large. All 
    original Messenger descriptions combined should take < 2GB of GPU memory.
    '''
    def __init__(self, model, tokenizer, device: torch.device, max_length:int=36):
        self.encoder = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length # max sentence length
        self.cache = {}

    def to(self, device):
        self.device = device
        self.encoder = self.encoder.to(device)
        return self

    def tokens_to_device(self, tokens):
        tok_device = {}
        for key in tokens:
            tok_device[key] = tokens[key].to(self.device)
        return tok_device

    def encode(self, text):
        '''
        Encodes the text using self.encoder and self.tokenizer. Text should be
        a list of sents, where sent is a string.
        '''
        encoded = [] # the final encoded texts
        for sent in text:
            if sent in self.cache.keys(): # sentence is in cache
                encoded.append(self.cache[sent])
            else: 
                with torch.no_grad():
                    tokens = self.tokenizer(
                        sent,
                        return_tensors="pt",
                        truncation=False,
                        truncation_strategy='do_not_truncate',
                        padding="max_length",
                        max_length=self.max_length
                    )
                    emb = self.encoder(**self.tokens_to_device(tokens)).last_hidden_state
                encoded.append(emb)
                self.cache[sent] = emb
        return torch.cat(encoded, dim=0)

def nonzero_mean(emb):
    '''
    Takes as input an embedding, emb. It should be H x W x L x D. with
    optional batch dimension. (H,W) is the grid dim, L the layers and
    D the embedding dimension. Returns mean of non-zero vectors along L dim.
    This is used to take care of overlapping sprites.
    '''
    # Count the number of non-zero vectors
    non_zero = torch.sum(torch.norm(emb, dim=-1) > 0, dim=-1)
    non_zero = non_zero.unsqueeze(-1).float()  # broadcasting
    non_zero[non_zero == 0] = 1 # prevent division by zero
    return torch.sum(emb, dim=-2) / non_zero