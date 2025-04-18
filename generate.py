from model import WordRNN
from dataset import stoi, itos, vocab_size
import torch

model = WordRNN(vocab_size)
model.load_state_dict(torch.load("word_model.pth"))
model.eval()

def generate(model, start_text, length=50):
    words = start_text.split()
    input_seq = torch.tensor([stoi[w] for w in words if w in stoi], dtype=torch.long).unsqueeze(0)
    hidden = None
    output = words[:]

    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_seq, hidden)
            probs = torch.softmax(logits[0, -1], dim=0)
            next_word_idx = torch.multinomial(probs, 1).item()
            next_word = itos[next_word_idx]
            output.append(next_word)
            input_seq = torch.tensor([[next_word_idx]])
    return ' '.join(output)

if __name__ == "__main__":
    print(generate(model, "Привет! ", length=30))