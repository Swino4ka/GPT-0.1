from model import CharRNN
from dataset import data, vocab_size
import torch
import torch.nn as nn

model = CharRNN(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.CrossEntropyLoss()

seq_length = 20
batch_size = 16
n_epochs = 100

for epoch in range(n_epochs):
    total_loss = 0
    for i in range(0, len(data) - seq_length, seq_length):
        x = data[i:i+seq_length]
        y = data[i+1:i+seq_length+1]
        x = x.unsqueeze(0)  
        y = y.unsqueeze(0)

        logits, _ = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")