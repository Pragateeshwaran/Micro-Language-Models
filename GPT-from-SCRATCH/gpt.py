import os 
import torch
import torch.nn as nn 
import torch.nn.functional as F


stories = ''
data_dir = r'F:\works\A-important\A-neurals\Vortex-Language-Models\GPT-from_SCRATCH\Data'
for story in os.listdir(data_dir):
    with open(os.path.join(data_dir, story), 'r', encoding='utf-8') as file:
        stories += file.read()
# print(stories[:256])

vocab = sorted(set(stories))
itos = {i: j for i, j in enumerate(vocab)}
stoi = {j: i for i, j in enumerate(vocab)}

def encoder(s):
    res = []
    for i in s:
        res.append(stoi[i])
    return res

def decoder(s):
    res = []
    for i in s:
        res.append(itos[i])
    return ''.join(res)


data = torch.tensor(encoder(stories))
train_size = int(len(data) * 0.9)

train_data = data[:train_size]
val_data = data[train_size:] 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vocab_size = 93
embedding_dimension = 100

batch_size = 16
block_size = 32

num_head = 4

N = 4

class HEAD(nn.Module):
    def __init__(self, head_size):
        super().__init__()        
        self.key = nn.Linear(embedding_dimension, head_size, bias=False)
        self.query = nn.Linear(embedding_dimension, head_size, bias=False)
        self.value = nn.Linear(embedding_dimension, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):    
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x)  
        wei = q @ k.transpose(-2,-1) * C**-0.5 # 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1)  
        v = self.value(x)  
        out = wei @ v  
        return out

class MULTIHEAD(nn.Module):
    def __init__(self, num_head, head_size):
        super().__init__()
        self.head = nn.ModuleList([HEAD(head_size) for i in range(num_head)]) 
        self.linear = nn.Linear(embedding_dimension, embedding_dimension)
        
    def forward(self, x):
        self.x = torch.cat([H(x) for H in self.head], dim= -1)
        self.x = self.linear(self.x)
        return self.x
    

class FEEDFORWARD(nn.Module):
    def __init__(self, embedding_dimension):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension*10), 
            nn.ReLU(),
            nn.Linear(10*embedding_dimension, embedding_dimension)
        )
        
    def forward(self, x):
        return self.layer(x)
    
class BLOCKS(nn.Module):
    def __init__(self, embedding_dimension, num_head):
        super().__init__()
        self.head_size = embedding_dimension // num_head
        self.attention_module = MULTIHEAD(num_head, self.head_size)
        self.feed_forward_module = FEEDFORWARD(embedding_dimension)
        self.layer_norm = nn.LayerNorm(embedding_dimension)
    def forward(self, x):
        # print("-----------------------")
        # print(x)
        x = x + self.attention_module(self.layer_norm(x))
        x = x + self.feed_forward_module(self.layer_norm(x))
        return x 

class shell_of_GPT(nn.Module):
    def __init__(self, vocab_size = vocab_size, embedding_dimension = embedding_dimension):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.positional_embedding = nn.Embedding(block_size, embedding_dimension)
        self.BLOCK = nn.Sequential(*[BLOCKS(embedding_dimension, num_head) for _ in range(N)])
        self.layer_norm = nn.LayerNorm(embedding_dimension)
        self.final_linear = nn.Linear(embedding_dimension, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx) 
        pos_emb = self.positional_embedding(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb
        # print("-------------------------------------")
        # print(x)
        x = self.BLOCK(x) 
        x = self.layer_norm(x)
        logits = self.final_linear(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
model = shell_of_GPT().to(device)


learning_rate = 1e-3
max_iters = 10000
eval_interval = 100
eval_iters = 200

batch_size = 4 
context_lenght = 8


def get_batch(split,):
    split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_lenght, (4, ))
    x = torch.stack([data[i : i+context_lenght] for i in ix])
    y = torch.stack([data[i + 1 : 1+i+context_lenght] for i in ix])
    return x.to(device), y.to(device)




optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
 
    xb, yb = get_batch('train')
 
    logits, loss = model(xb, yb)
    if iter % 100 == 0:
        print(f"the loss at iteraton iter is : {loss}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


