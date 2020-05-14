#%%
import sys
from tqdm import tqdm
import torch
import torch.nn as nn

sys.path.append("..")
from models.TransformerLM import TransformerLM
from utils.data import *
from utils.train import *

batch_size = 32
eval_batch_size = 32

train_txt, val_txt, test_txt, TEXT = get_wikitext_splits()

train_data = batchify(train_txt, batch_size, TEXT)
val_data = batchify(val_txt, eval_batch_size, TEXT)
test_data = batchify(test_txt, eval_batch_size, TEXT)

bptt = 35
n_tokens = len(TEXT.vocab.stoi)
emb_sz = 200
n_hid = 200
n_layers = 2
n_attn_heads = 2
dropout = 0.2

model = TransformerLM(n_tokens, emb_sz, n_attn_heads, n_hid, n_layers, dropout)
model = model.to(device)

crit = nn.CrossEntropyLoss()
lr = 5.0
optim = torch.optim.SGD(model.parameters(), lr=lr)
sched = torch.optim.lr_scheduler.StepLR(optim, 1.0, gamma=0.95)

#%%
best_val_loss = float("inf")
epochs = 10
best_model = None

for epoch in range(1, epochs + 1):
    train(model, crit, optim, lr, sched, epoch, train_data, TEXT.vocab, bptt)
    sched.step()
# %%
