"""
To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py

- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)

"""
import sys
sys.path.insert(1, '../')
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from data.hellaswag import render_example, iterate_examples

# --------------------------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
  '''The input consists of
  queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the
  query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the
  values.
  In practice, we compute the attention function on a set of queries simultaneously, packed together
  into a matrix Q. The keys and values are also packed together into matrices K and V . We compute
  the matrix of outputs as:
  Attention(Q,K,V ) = softmax(QKT/√dk)V'''

  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    
    # key, query, value projections for all heads, but in a batch
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

    # output projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1 # for residual init

    # regularization
    self.n_head = config.n_head
    self.n_embd = config.n_embd

    # not really a 'bias' more of a mask, but following the openAI/HF naming though
    # self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    B, T, C = x.size() # batch size, sequence length and embedding dimensionality (n_embd)

    # calculate query, key, values for all heads in batch and move head forward to match the batch dim
    # nh is 'number of heads', hs is 'head size' and C (number of channels) = nh * hs
    # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
    # We have tokens lined up in a sequence of 1024 tokens and each of them emit 3 vectors query, key and value
    # So in the line below what happens is we multiply queries keys and values to get the attention ot affinities
    qkv = self.c_attn(x)
    
    q, k, v = qkv.split(self.n_embd, dim=2) # we split into q, k , v

    # Here we have included number of heads as a batch dimension so we do not have the explicit for loop to concat the output of each heads
    # so B and nh together now kind of act like a batch/batches
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

    # attention materializes the large (T, T) matrix for all queries and key
    # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
    # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # (B, nh, T, T) - decoder
    # att = F.softmax(att, dim=-1) # (B, nh, T, T)
    # y = att @ v # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs) weighted sum of the tokens we found interesting

    # Replacing the above lines for attention with flash attention
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # is causal is also known as masked attention
    # is_causal = True -> This ensures that a token at a given position in a sequence can only attend to tokens that appeared at or before its current position, 
    # and not to any future tokens

    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, actual concatenation operation is happening here

    # output projection
    y = self.c_proj(y)
    return y


class MLP(nn.Module): # feed- forward - FFN(x) = max(0,xW1 + b1)W2 + b2 
  '''In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
  connected feed-forward network, which is applied to each position separately and identically. This
  consists of two linear transformations with a ReLU activation in between'''

  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd,  4 * config.n_embd) # we always have the feedforward layer four times the size of the embd layer, dff = 4 * dmodel)
    self.gelu = nn.GELU(approximate='tanh') # GPT2 uses the approximate version
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1 # for residual init

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x


class Block(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config) # communication is done by self attention, tokens communicate with each other/ aggregation fn /reduce fn
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config) # Feed-Forward Networks - MLP works on every single token, essentially a map function

  
  def forward(self, x):
    x = x + self.attn(self.ln_1(x)) # x + -> residual connection
    x = x + self.mlp(self.ln_2(x))  # x + -> residual connection
    return x

@dataclass
class GPTConfig:
  block_size: int = 1024        # 1024   -> max sequence length
  vocab_size: int = 50257       # 50257  -> 256: raw byte tokens + 50,000 merges done by openAI + 1 special token(<|endoftext|>)
  n_layer: int = 12             # 12     -> number of layers
  n_head: int = 12              # 12     -> number of heads
  n_embd: int = 768             # 768    -> embedding dimension

class GPT(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.config = config

    self.transformer = nn.ModuleDict(dict(
      wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
      wpe = nn.Embedding(config.block_size, config.n_embd), # position embedding
      h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # no of transformer blocks
      ln_f = nn.LayerNorm(config.n_embd) # GPT2-paper: final layer norm after the final self-attention block
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # n_embd to vocab size - final linear layer

    # weight sharing scheme
    self.transformer.wte.weight = self.lm_head.weight

    # https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.apply
    self.apply(self._init_weights)

  
  def _init_weights(self, module):
    
    if isinstance(module, nn.Linear):
      std = 0.02

      if hasattr(module, 'NANOGPT_SCALE_INIT'):
        std *= (2 * self.config.n_layer) ** -0.5 # residual init we are multiplying 2 because we have 2 residual connections self.attn and self.mlp

      # https://github.com/openai/gpt-2/blob/master/src/model.py - line number 152 - 155
      # https://docs.pytorch.org/docs/2.9/nn.init.html#torch.nn.init.normal_
      torch.nn.init.normal_(module.weight, mean=0.0, std=std)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
      
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


  def forward(self, idx, targets=None):
    # idx is of shape (B, T)
    B, T = idx.size()
    assert T <= self.config.block_size, f'Cannot forward sequence of length {T}, block size is only {self.config.block_size}'

    # forward the token and positional embeddings
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
    pos_emb = self.transformer.wpe(pos) # shape (T, n_embd)
    tok_emb = self.transformer.wte(idx) # shape (B, T, n_embd)
    x = tok_emb + pos_emb # shape (B, T, n_embd)
    # forward to the block of the transformer
    for block in self.transformer.h:
       x = block(x)

    # forward to the final layer norm and classifier
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x) # (B, T, vocab_size)
    loss = None

    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # reduction is by default mean
  
    return logits, loss
  
  def configure_optimizers(self, weight_decay, learning_rate, device_type):
    # start with all of the candidate parameters that requires grad
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # create optim groups. Any parameter that is 2D will be weight decayed otherwise no
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layer norms don't
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
      {'params': decay_params, 'weight_decay': weight_decay},
      {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)

    if master_process:
      print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
      print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    use_fused = device_type == 'cuda'
    # create AdamW optimizer 
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer


  @classmethod
  def from_pretrained(cls, model_type):
    '''Loads pretrained GPT-2 model weights from hugging face'''
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    from transformers import GPT2LMHeadModel
    print('loading weights from pretrained gpt: %s' % model_type)

    # n_layer, n_head and n_embd are determined from model_type
    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[model_type]
    config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
    # create a from-scratch initialized minGPT model
    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

    # init a huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()
    # copy while ensuring all of the parameters are aligned and match in names and shapes
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    # basically the openai checkpoints use a 'Conv1D' module, but we only want to use a vanilla Linear
    # this means that we have to transpose these weights when we import them
    assert len(sd_keys_hf) == len(sd_keys), f'mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}'
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            # vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])
    return model

# --------------------------------------------------------------------------------------------------------
import tiktoken
import numpy as np

enc = tiktoken.get_encoding('gpt2')

def load_tokens(filename):
  npt = np.load(filename)
  ptt = torch.tensor(npt, dtype=torch.long)
  return ptt


class DataLoaderLite:
   
  def __init__(self, B, T, process_rank, num_processes, split):
    self.B = B
    self.T = T
    self.process_rank = process_rank
    self.num_processes = num_processes
    assert split in { 'train', 'val'}


    # # at init load tokens from disk and load in the memory
    # with open('./data/input.txt', 'r') as f:
    #   text = f.read()
  
    # enc = tiktoken.get_encoding('gpt2')
    # tokens = enc.encode(text, allowed_special={'<|endoftext|>'})
    # self.tokens = torch.tensor(tokens)

    # if master_process:
    #   print(f'loaded {len(self.tokens)} tokens')
    #   print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

    # get the shard filenames
    # data_root = '/content/drive/MyDrive/nanoGPT/edu_fineweb10B' # for google colab
    data_root = '../data/edu_fineweb10B'
    shards = os.listdir(data_root)
    shards = [s for s in shards if split in s]
    shards = sorted(shards)
    shards = [os.path.join(data_root, s) for s in shards]
    self.shards = shards
    assert len(shards) > 0, f'no shards found for split {split}'

    if master_process:
      print(f'found {len(shards)} shards for split {split}')
    
    self.reset()

  def reset(self):
     # state, init at shard zero
    self.current_shard = 0
    self.tokens = load_tokens(self.shards[self.current_shard])
    # process_rank: 0 --> starts at 0
    # process_rank: 1 --> starts at 1 * B * T
    # process_rank: 2 --> starts at 2 * B * T
    self.current_position = self.B * self.T * self.process_rank

  
  def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_position: self.current_position + B * T + 1]
    x = buf[:-1].view(B, T) # inputs
    y = buf[1:].view(B, T) # targets

    # advance the position of the tensor
    self.current_position += B * T * self.num_processes

    # if loading the next batch is out of bounds, advance to next shard
    if self.current_position + ( B * T * self.num_processes + 1) > len(self.tokens):
      # self.current_position = self.B * self.T * self.process_rank
      self.current_shard = (self.current_shard + 1) % len(self.shards)
      self.tokens = load_tokens(self.shards[self.current_shard])
      self.current_position = B * T * self.process_rank

    return x, y

# --------------------------------------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
  # shape of tokens and mask for example say (4, 21)
  # evaluate the autoregressive loss at all positions
  shift_logits = (logits[..., :-1, :]).contiguous() # (B, T-1, vocab_size)
  shift_tokens = (tokens[..., 1:]).contiguous() # (4, 20)
  flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)) # (B * (T-1), vocab_size)
  flat_shift_tokens = shift_tokens.view(-1) # (80)

  shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none') # (80)
  shift_losses = shift_losses.view(tokens.size(0), -1) # (4, 20)
  # now get the average loss just for the completion region (where mask == 1), in each row
  shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token # (4, 20)
  masked_shift_losses = shift_losses * shift_mask
  # sum and divide by the number of 1s in the mask
  sum_loss = masked_shift_losses.sum(dim=1)
  avg_loss = sum_loss / shift_mask.sum(dim=1)
  # now we have a loss for each of the 4 completions
  # the one with the lowest loss should be the most likely
  pred_norm = avg_loss.argmin().item()
  return pred_norm

# --------------------------------------------------------------------------------------------------------
import time
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (Distributed Data Parallel)
# torchrun command sets the environment variables RANK, LOCAL_RANK and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a DDP run?
if ddp:
  if torch.cuda.is_available():
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
  else:
    init_process_group(backend='gloo')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cpu:{ddp_local_rank}'
    print(f'{ddp_rank=}, {ddp_local_rank=}, {ddp_world_size=}, {device=}')
    torch.cpu.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
  # vanilla non-ddp run
  ddp_rank = 0
  ddp_local_rank = 0
  ddp_world_size = 1
  master_process = True
  # attempt to autodetect device
  device = "cpu"
  if torch.cuda.is_available():
      device = "cuda"
  elif hasattr(torch.backends, "mps") and torch.mps.is_available():
      device = "mps"
  print(f"using device: {device}")

# --------------------------------------------------------------------------------------------------------

torch.manual_seed(1337)

# auto detect device
if torch.cuda.is_available():
   torch.cuda.manual_seed(1337)

elif hasattr(torch.backends, 'mps') and torch.mps.is_available():
  torch.mps.manual_seed(1337)

# print(f'using device: {device}')

# --------------------------------------------------------------------------------------------------------

# Model Name    nparams nlayers dmodel nheads dhead Batch Size Learning Rate
# GPT-3 Small     125M      12    768     12    64      0.5M     6.0 ×10−4
# we can see from the GPT-3 paper and the table above that the batch size is 0.5M, we cannot just increase batch size to 0.5M
# To achieve this we are going to use gradient accumulation

total_batch_size = 524288 # 2**19, ~0.5M in terms of tokens
B = 16 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, 'make sure total batch size is divisible by B * T * ddp_world_size'
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
  print(f"total desired batch size: {total_batch_size}")
  print(f"calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

# --------------------------------------------------------------------------------------------------------
# https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision('high')


# create model
model = GPT(GPTConfig(vocab_size=50304)) # nice number powers of 2
model.to(device)
use_compile = False

if use_compile:
  model = torch.compile(model) # does not work with mps on apple - issue: https://github.com/pytorch/pytorch/issues/171764
# model = torch.compile(model) # does not work with mps on apple - issue: https://github.com/pytorch/pytorch/issues/171764

if ddp:
  if device == 'cuda':
    model = DDP(model, device_ids=[ddp_local_rank])
  
  else:
    model = DDP(model)

raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# learning rate
# GPT-3 paper - B Details of Model Training
# we use cosine decay for learning rate down to 10% of its value, over 260 billion tokens 
# (after 260 billion tokens, training continues at 10% of the original learning rate). 
max_lr = 6e-4
min_lr = 0.1 * max_lr # 10 % of max lr
warmup_steps = 715 # total batch size is 2^19, GPT 3 paper says they warmup to 375 million tokens, so 375e6 / 2**19 = 715 steps
max_steps = 19073  # total batch size is 2^19, we have 10B tokens, so 10e9 / 2**19 = 19073 steps

def get_lr(it):

  # 1) Linear warm up for warmup_iterations
  if it < warmup_steps:
    return max_lr * (it + 1) / warmup_steps
  
  # 2) if it > lr_decay_iters return min learning rate
  if it > max_steps:
    return min_lr

  # 3) in between use cosine decay down to min learning rate
  decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
  return min_lr + coeff * (max_lr - min_lr)


# Optimization
# GPT-3 paper - B Details of Model Training
# To train all versions of GPT-3, we use Adam with β1 = 0.9, β2 = 0.95, and  = 10−8
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate = 6e-4, device_type=device)

# create a log directory we will write checkpoints to and log to
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'log.txt')
with open(log_file, 'w') as f: # open for writing to clear the file
  pass


# def save_checkpoint(state, filename):
#   "save the checkpoint"
  

for step in range(max_steps):
  t0 = time.time()
  last_step = (step == max_steps - 1)

  # once in a while evaluate our validation loss
  if step % 250 == 0 or last_step:
    model.eval()
    val_loader.reset()

    with torch.no_grad():
      val_loss_accum = 0.0
      val_loss_steps = 20
      for _ in range(val_loss_steps):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
          logits, loss = model(x, y)
        
        loss = loss / val_loss_steps
        val_loss_accum += loss.detach()
    
    if ddp:
      dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    
    if master_process:
      print(f'validation loss: {val_loss_accum.item():.4f}')
      with open(log_file, 'a') as f:
        f.write(f"{step} val {val_loss_accum.item():.4f}\n")
      
      if step > 0 and (step % 5000 == 0 or last_step):
        # optionally write model checkpoints
        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        checkpoint = {
          'model': raw_model.state_dict(),
          'config': raw_model.config,
          'step': step,
          'val_loss': val_loss_accum.item()
        }
        # you might also want to add optimizer.state_dict() and
        # rng seeds etc., if you wanted to more exactly resume training
        torch.save(checkpoint, checkpoint_path)

  
  # once in a while evaluate hellaswag
  if (step % 250 == 0 or last_step) and (not use_compile):
    num_correct_norm = 0
    num_total = 0

    for i, example in enumerate(iterate_examples('val')):
      # only process examples where i % ddp_world_size == ddp_rank
      if i % ddp_world_size != ddp_rank:
        continue

      # render the example into tokens and labels
      _, tokens, mask, label = render_example(example)
      tokens = tokens.to(device)
      mask = mask.to(device)

      # get the logits
      with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
          logits, loss = model(tokens)
        
        pred_norm = get_most_likely_row(tokens, mask, logits)
      
      num_total += 1
      num_correct_norm += int(pred_norm == label)
    
    # reduce the stats across all processes
    if ddp:
      num_total = torch.tensor(num_total, dtype=torch.long, device=device)
      num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
      dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
      dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
      num_total = num_total.item()
      num_correct_norm = num_correct_norm.item()

    acc_norm = num_correct_norm / num_total

    if master_process:
      print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
      with open(log_file, "a") as f:
        f.write(f"{step} hella {acc_norm:.4f}\n")

  # once in a while generate from the model (except step 0, which is noise)
  # disable torch.compile - throws scary errors and do not work on my mac as 'mps' has a lot of issues with torch.compile
  if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
    model.eval()
    num_return_sequences = 4
    max_length = 32
    tokens = enc.encode('Hello, I am a language model,') # generates 8 tokens -> [15496, 11, 314, 716, 257, 3303, 2746, 11]
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (4, 8)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)

    while xgen.size(1) < max_length:
      # forward the model to get the logits
      with torch.no_grad():
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
          logits, loss = model(xgen) # (B, T, vocab_size)

        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        xgen = torch.cat((xgen, xcol), dim=1)
      
    # print the generated text
    for i in range(num_return_sequences):
      tokens = xgen[i, :max_length].tolist()
      decoded = enc.decode(tokens)
      print(f"rank {ddp_rank} sample {i}: {decoded}")


  # train loop
  model.train()
  optimizer.zero_grad()
  loss_accum = 0.0

  for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    # https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html - very important
    # https://docs.pytorch.org/docs/stable/amp.html#torch.autocast - very important
    with torch.autocast(device_type=device, dtype=torch.bfloat16): # with bfloat16 we need not perform any gradient scaling
      logits, loss = model(x, y)
      # import code; code.interact(local=locals())
    
    # we have to scale the loss to account for gradient accumulation
    # because the gradients just add on each successive backward().
    # addition of these gradients corresponds to a SUM in the objective, but
    # instead of SUM we want a MEAN. Scale the loss here so it comes out right
    loss = loss / grad_accum_steps
    loss_accum += loss.detach()

    if ddp:
      # https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync
      # https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/parallel/distributed.py#L1436 - self.require_backward_grad_sync = False
      model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # basically we want to sync at the last step only

    loss.backward() # so this will sync at the very last step as require_backward_grad_sync will be true only for the last step
    print(f'microstep: {micro_step} | loss_accum: {loss_accum.item():.6f} completed')

  if ddp:
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

  # GPT-3 paper - B Details of Model Training
  # we clip the global norm of the gradient at 1.0
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

  # determine the learning rate for this iteration
  lr = get_lr(step)

  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

  optimizer.step()
  
  if device == 'cuda':
    torch.cuda.synchronize() # Waits for all kernels in all streams on a CUDA device to complete.
  elif device == 'mps':
    torch.mps.synchronize() # Waits for all kernels in all streams on a MPS device to complete.

  t1 = time.time()
  dt = (t1 - t0)
  tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
  tokens_per_sec = tokens_processed / dt

  if master_process:
    print(f'step: {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}')
    with open(log_file, 'a') as f:
      f.write(f'{step} train {loss_accum.item():.6f} \n')


if ddp:
  destroy_process_group()

# --------------------------------------------------------------------------------------------------------