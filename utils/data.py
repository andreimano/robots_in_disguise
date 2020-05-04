# As seen on https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import torch
import torchtext
from torchtext.data.utils import get_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_wikitext_splits():
    TEXT = torchtext.data.Field(
        tokenize=get_tokenizer("basic_english"),
        init_token="<sos>",
        eos_token="<eos>",
        lower=True,
    )

    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_txt)
    return train_txt, val_txt, test_txt, TEXT


def batchify(data, batch_sz, TEXT):
    data = TEXT.numericalize([data.examples[0].text])
    n_batch = data.size(0) // batch_sz
    data = data.narrow(0, 0, n_batch * batch_sz)
    data = data.view(batch_sz, -1).t().contiguous()
    data = data.to(device)
    return data
