import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

# hyperparameters -------
batch_size = 64 # how many independent sequences will we process in parallel
block_size = 256 # what is the maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2 # 20% is dropped
# hyperparameters -------

with open('./data/shakespeare/input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integer
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# let's now split the data into train and val sets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
  # generate small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i: i + block_size] for i in ix]) # (32, 8) -> stack by row to make it a (batch_size, block_size) vector
  y = torch.stack([data[i + 1: i + block_size + 1] for i in ix]) # (32, 8) -> stack by row to make it a (batch_size, block_size) vector
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
  out = {}
  # model.eval() is a method in PyTorch that switches a neural network to evaluation (inference) mode. 
  # This is crucial because certain layers, such as Dropout and BatchNorm, behave differently during training and inference
  model.eval()

  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    
    out[split] = losses.mean()
  
  model.train() # resetting to train phase
  return out

class Head(nn.Module):
  """one head of self attention"""

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    # The register_buffer method in PyTorch is used within an nn.Module to store a tensor that is part of the model's persistent state 
    # but should not be updated by the optimizer during training. 
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x) # (B, T, C)
    q = self.query(x) # (B, T, C)

    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T) # scaled attention
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) # decoder block - future doesn't communicate with the past
    wei = F.softmax(wei, dim=-1) # (B, T, T)
    wei = self.dropout(wei)

    # perform the weighted aggregation of the values
    v = self.value(x) # (B, T, C)
    out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
    return out
  
class MultiHeadAttention(nn.Module):
  """ multiple heads of self attention in parallel"""

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    
    out =  torch.cat([h(x) for h in self.heads], dim=-1) # concatenating over the channel dimension
    out = self.dropout(self.proj(out))
    return out
  
  

class FeedForward(nn.Module):
  """a simple linear layer followed by non-linearity"""

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd), # read the paper 3.3 - Position-wise Feed-Forward Networks
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd), # projection layer going back into the residual pathway
      nn.Dropout(dropout), # dropout is added just before the connection back into the residual pathway
    )
  
  def forward(self, x):
    return self.net(x)
  

class Block(nn.Module):
  """ Transformer block: communication followed by computation """

  def __init__(self, n_embd, n_head):
    # n_embd: embedding dimension, n_head: the number of heads we'd like
    super().__init__()
    head_size = n_embd // n_head # size 32 // 4 = 8

    # i.e. 4 heads of 8-dimensional self-attention - gathers information about the current token w.r.t to the previous tokens
    self.sa = MultiHeadAttention(n_head, head_size) # communication is done by multi-headed self attention

    # thinks on the gathered info by the self attention layer
    self.ffwd = FeedForward(n_embd) # computation is done by the feed forward layer

    # layer normalizations
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x)) # residual connections, layer norm is applied directly on x before it goes on to self attention and feed forward
    x = x + self.ffwd(self.ln2(x)) # residual connections
    return x


# super simple bi-gram model
class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd) # so each position from zero to block size -1 will also get its own embedding vector
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm before the final linear layer
    self.lm_head = nn.Linear(n_embd, vocab_size) # language modelling head


  def forward(self, idx, targets=None): # idx --> x, targets --> y (32, 8)

    # idx and targets are both (B, T) tensor of integers[B -> batch_size, T -> block_size]
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx) # (B, T, C) # B - Block - batch_size, T - Time - block_size, C - Channel
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
    x = tok_emb + pos_emb # (B, T, C) -> broadcasting
    # x = self.sa_head(x) # apply one head of self attention. (B, T, C)
    # x = self.ffwd(x) # (B, T, C)
    x = self.blocks(x)
    logits = self.lm_head(x) # (B, T, vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B * T, C) # re-shaping to match pytorch cross entropy definition
      targets = targets.view(B * T)
      loss = F.cross_entropy(logits, targets)
    
    return logits, loss
  
  def generate(self, idx, max_new_tokens):

    # idx is (B, T) array of indices in current context
    for _ in range(max_new_tokens):
      # crop idx to the last block_size tokens
      idx_cond = idx[:, -block_size:] # positional embedding will run out of scope

      # get the predictions
      logits, loss = self(idx_cond) # shape of logits (1, 1, 65), (1, 2, 65)....(1, max_new_tokens, 65)
      
      # focus only on the last time step
      logits = logits[:, -1, :] # becomes (B, C)

      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=1) # (B, C)

      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

      # append the sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T + 1) (1), (1, 1), (1, 2)...
    
    return idx
  
model = BigramLanguageModel()
m = model.to(device)

# create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for iter in range(max_iters):
  # every once in a while evaluate the loss on train and val sets
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')
  # sample a batch of data
  xb, yb = get_batch('train')
  # evaluate loss
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

# generate from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=1500)[0].tolist()))


