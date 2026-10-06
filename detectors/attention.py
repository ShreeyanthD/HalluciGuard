"""CLAP-style projected layer sequence, CLS token, Transformer encoder.

Architectural baseline only: pooling/token position and training recipe are
specified here and are not a claim to reproduce published CLAP numbers.
"""
import numpy as np


class AttentionProbe:
    def __init__(self, seed=0, steps=100, width=32, lr=0.002):
        self.seed, self.steps, self.width, self.lr = seed, steps, width, lr

    def fit(self, x, y, groups):
        import torch
        from torch import nn
        if set(np.asarray(y).tolist()) != {0, 1}:
            raise ValueError('Probe training requires both classes')
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        self.mean = x.mean(0)
        self.scale = np.maximum(x.std(0), 1e-5)
        width = self.width
        class Network(nn.Module):
            def __init__(net):
                super().__init__()
                net.proj = nn.Linear(x.shape[-1], width)
                net.cls = nn.Parameter(torch.zeros(1, 1, width))
                net.pos = nn.Parameter(torch.zeros(1, x.shape[1] + 1, width))
                net.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                    width, 4, width * 2, dropout=0.0, batch_first=True), 1,
                    enable_nested_tensor=False)
                net.head = nn.Linear(width, 1)
            def forward(net, inputs):
                seq = torch.cat([net.cls.expand(len(inputs), -1, -1), net.proj(inputs)], 1)
                return net.head(net.encoder(seq + net.pos)[:, 0]).squeeze(-1)
        self.net = Network()
        inputs = torch.tensor((x - self.mean) / self.scale, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.float32)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=0.01)
        for _ in range(self.steps):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(self.net(inputs), labels)
            loss.backward()
            opt.step()
        self.net.eval()
        return self

    def predict(self, x):
        import torch
        with torch.no_grad():
            return self.net(torch.tensor((x - self.mean) / self.scale, dtype=torch.float32)).sigmoid().numpy()
