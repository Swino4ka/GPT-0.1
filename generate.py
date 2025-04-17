import torch
from dataset import itos
from dataset import encode
def generate(model, start_text, length=100):
    model.eval()
    input_seq = torch.tensor(encode(start_text), dtype=torch.long).unsqueeze(0)
    hidden = None
    output_text = start_text

    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_seq, hidden)
            probs = torch.softmax(logits[0, -1], dim=0)
            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = itos[next_char_idx]
            output_text += next_char
            input_seq = torch.tensor([[next_char_idx]])
    return output_text