import torch
import torch.nn as nn
import datetime
from model import MinerFormer

# ==== Confifuration ====
device = "cuda" if torch.cuda.is_available() else "cpu"
intervals = ["1m", "5m", "1d"]
bars = len(intervals)
B, T, D = 2, 127, 5  # Batch, Sequence Length, Features (OHLCV)
LLM_DIM = 64

# ==== Model Initialization ====
model = MinerFormer(
    intervals=bars,
    open_idx=0, close_idx=3, highest_idx=1, lowest_idx=2,
    dimmensions=D, llm_emebed_dim=LLM_DIM,
    max_seq_len=T, embed_dim=32, num_heads=2,
    ff_dim=64, num_layers=2, dropout=0.1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# ==== Fake Kline data ====
def make_fake_kline(base=30000, volatility=0.02, steps=127):
    data = []
    for _ in range(steps):
        open_price = base * (1 + torch.randn(1).item() * volatility)
        high = open_price * (1 + torch.rand(1).item() * 0.01)
        low = open_price * (1 - torch.rand(1).item() * 0.01)
        close = base * (1 + torch.randn(1).item() * volatility)
        volume = torch.rand(1).item() * 100
        data.append([open_price, high, low, close, volume])
    return torch.tensor(data)

inputs_seq = []
for _ in intervals:
    batch_bar = []
    for _ in range(B):
        batch_bar.append(make_fake_kline())
    inputs_seq.append(torch.stack(batch_bar).to(device))  # (B, T, D)

x_target = []
for b in range(B):
    o = 30000 * (1 + torch.randn(1).item() * 0.02)
    h = o * 1.01
    l = o * 0.99
    c = 30000 * (1 + torch.randn(1).item() * 0.02)
    v = torch.rand(1).item() * 100
    x_target.append([o, h, l, c, v])
x_target = torch.tensor(x_target).unsqueeze(1).to(device)

# ==== Fake llm embedding vector ====
llm_vector = torch.randn(B, LLM_DIM, device=device)
llm_embedding = llm_vector.unsqueeze(1).expand(-1, T, -1)

# ==== single step training ====
model.train()
preds = model(inputs_seq, llm_embedding)
pred_last = preds[0][:, -1, :]
target_last = x_target[:, 0, [1, 2]]

loss = loss_fn(pred_last, target_last)

optimizer.zero_grad()
loss.backward()
optimizer.step()

now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"[{now}] ✅ Demo Step | Loss = {loss.item():.6f}")
print(f"Predicted High/Low: {pred_last[0].tolist()}")
print(f"Target High/Low   : {target_last[0].tolist()}")

