from model import WordRNN
from dataset import data, vocab_size
import torch
import torch.nn as nn
from tqdm import tqdm

model = WordRNN(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loss_fn = nn.CrossEntropyLoss()

SEQ_LEN = 10
EPOCHS = 200

for epoch in range(EPOCHS):
    total_loss = 0
    for i in range(0, len(data) - SEQ_LEN, SEQ_LEN):
        x = data[i:i+SEQ_LEN].unsqueeze(0)
        y = data[i+1:i+SEQ_LEN+1].unsqueeze(0)

        logits, _ = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "word_model.pth")