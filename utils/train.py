import torch
import torch.nn as nn
from tqdm import tqdm
import math


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def train(
    model, crit, optim, lr, sched, epoch, train_data, vocab, bptt, verbose=200,
):
    model.train()
    total_loss = 0.0
    n_tokens = len(vocab.stoi)

    train_bar = tqdm(range(0, train_data.size(0) - 1, bptt))

    for batch, i in enumerate(train_bar):
        data, targets = get_batch(train_data, i, bptt)
        optim.zero_grad()
        out = model(data)
        loss = crit(out.view(-1, n_tokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()

        total_loss += loss.item()
        log_interval = 200

        if batch % verbose == 0 and batch > 0:
            cur_loss = total_loss / verbose
            train_bar.set_description(
                f"epoch:{epoch} | lr: {sched.get_lr()[0]:.3f} | loss:"
                + f"{cur_loss}:.3f, perplexity: {math.exp(cur_loss):.3f}"
            )

            total_loss = 0


def eval(model, crit, data, vocab, bptt):
    model.eval()
    total_loss = 0.0
    n_tokens = len(vocab.stoi)
    with torch.no_grad():
        val_bar = tqdm(range(0, data.size(0) - 1, bptt))
        for i in val_bar:
            data, targets = get_batch(data, i)
            out = model(data)
            out_flat = out.view(-1, n_tokens)
            total_loss += len(data) * crit(out_flat, targets).item()
    return total_loss / (len(data) - 1)
