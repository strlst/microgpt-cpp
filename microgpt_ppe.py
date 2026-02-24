"""
Karpathy's `microgpt.py` adapted to use pytorch_pfn_extras (PPE) as the
training engine. The original implements a full GPT-2-like neural network
architecture and thus serves as a popular learning resource.
Please check out Karpathy's blog for more information:
http://karpathy.github.io/2026/02/12/microgpt/

@strlst (original by @karpathy)
"""

import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_pfn_extras as ppe
from torch.utils.data import Dataset, DataLoader
from typing import List, Set, Dict

# reproducibility
random.seed(42)
torch.manual_seed(42)

# hyperparameters
# scaled up a little to make proper use of hardware
n_embd = 32
n_head = 8
n_layer = 3
block_size = 16
head_dim = n_embd // n_head


def load_dataset() -> List[str]:
    if not os.path.exists("input.txt"):
        import urllib.request

        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt",
            "input.txt",
        )
    docs = [
        l.strip()
        for l in open("input.txt").read().strip().split("\n")
        if l.strip()
    ]
    random.shuffle(docs)
    return docs


class Tokenizer:
    def __init__(self, docs: List[str]):
        self.docs = docs
        self.uchars = sorted(set("".join(docs)))
        # generate BOS token id as a guaranteed to be unique id
        self.BOS = len(self.uchars)
        self.vocab_size = len(self.uchars) + 1

    def encode(self, doc: str) -> torch.Tensor:
        ids = [self.BOS] + [self.uchars.index(ch) for ch in doc] + [self.BOS]
        return torch.tensor(ids, dtype=torch.long)

    def as_name(self, tokens: List[int]) -> str:
        return "".join(self.uchars[t] for t in tokens[1:])


# enable iteration over (inputs, targets) pairs
class NamesDataset(Dataset):
    def __init__(self, split_docs: List[str], tokenizer: Tokenizer):
        self.samples = []
        for doc in split_docs:
            tokens = tokenizer.encode(doc)
            n = min(block_size, len(tokens) - 1)
            tokens = tokens[: n + 1]
            self.samples.append((tokens[:-1], tokens[1:]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inputs, targets = self.samples[idx]
        return inputs, targets


def padding(batch):
    # pad variable-length sequences in a batch
    inputs_list, targets_list = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs_list, batch_first=True)
    targets = torch.nn.utils.rnn.pad_sequence(
        targets_list, batch_first=True, padding_value=-100
    )
    return inputs, targets


# model modules
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        # (B, H, T, D)
        q = self.wq(x).view(B, T, n_head, head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, n_head, head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, n_head, head_dim).transpose(1, 2)

        scale = math.sqrt(head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        # (B, H, T, D)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

        self.wte = nn.Embedding(tokenizer.vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm = RMSNorm(n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, tokenizer.vocab_size, bias=False)

        nn.init.normal_(self.wte.weight, std=0.08)
        nn.init.normal_(self.wpe.weight, std=0.08)
        nn.init.normal_(self.lm_head.weight, std=0.08)
        for block in self.blocks:
            for p in block.parameters():
                nn.init.normal_(p, std=0.08)

    def _encode(self, tokens: torch.Tensor) -> torch.Tensor:
        # shared encoder: tokens (B, T) -> logits (B, T, vocab_size)
        B, T = tokens.shape
        # (1, T)
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        # (B, T, C)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm(x)
        for block in self.blocks:
            x = block(x)
        # (B, T, vocab_size)
        return self.lm_head(x)

    # PPE-compatible forward
    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # (B, T, V)
        logits = self._encode(inputs)
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.view(B * T, V),
            targets.view(B * T),
            ignore_index=-100,
        )

        prefix = "train" if self.training else "val"
        ppe.reporting.report({f"{prefix}/loss": loss.item()})

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self, temperature: float = 0.5, max_new_tokens: int = block_size
    ):
        self.eval()
        tokens = [self.tokenizer.BOS]
        for _ in range(max_new_tokens):
            # (1, T)
            idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            # (1, T, V)
            logits = self._encode(idx)
            next_logits = logits[0, -1] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            if next_token == self.tokenizer.BOS:
                break
            tokens.append(next_token)
        return self.tokenizer.as_name(tokens)


# ppe-based training
def make_loaders(
    tokenizer: Tokenizer, val_split=0.1, batch_size=32
):
    split = int(len(tokenizer.docs) * (1 - val_split))
    train_ds = NamesDataset(tokenizer.docs[:split], tokenizer)
    val_ds = NamesDataset(tokenizer.docs[split:], tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=padding
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=padding
    )
    return train_loader, val_loader


def train(
    model: GPT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    device: str = "cpu",
):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, betas=(0.85, 0.99), eps=1e-8
    )

    extensions = [
        ppe.training.extensions.LogReport(),
        ppe.training.extensions.ProgressBar(),
        ppe.training.extensions.PrintReport(
            ["epoch", "iteration", "train/loss", "val/loss", "elapsed_time"]
        ),
        # snapshot uses lowercase..
        ppe.training.extensions.snapshot(
            filename="snapshot_epoch_{.epoch}",
            n_retains=epochs,
            autoload=True,
        ),
    ]

    trainer = ppe.engine.create_trainer(
        model,
        optimizer,
        epochs,
        evaluator=ppe.engine.create_evaluator(
            model,
            device=device,
            progress_bar=True,
        ),
        device=device,
        extensions=extensions,
    )

    ppe.to(model, device)
    trainer.run(train_loader, val_loader)


def inference(model: GPT, samples: int = 30, temperature: float = 0.5):
    print("\n--- inference (hallucinated names) ---")
    for i in range(samples):
        name = model.generate(temperature=temperature)
        print(f"sample {i+1:2d}: {name}")


def main():
    docs = load_dataset()
    print(f"num docs: {len(docs)}")

    tokenizer = Tokenizer(docs)
    print(f"vocab size: {tokenizer.vocab_size}")

    model = GPT(tokenizer)
    print(f"num params: {sum(p.numel() for p in model.parameters())}\n")

    train_loader, val_loader = make_loaders(
        tokenizer, val_split=0.1, batch_size=32
    )

    # run training
    train(
        model,
        train_loader,
        val_loader,
        epochs=3,
        device="cpu",
    )

    # finally run inference
    inference(model)


if __name__ == "__main__":
    main()
