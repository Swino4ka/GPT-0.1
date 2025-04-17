with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

words = text.split()  # простейшая токенизация по пробелам
vocab = sorted(set(words))
vocab_size = len(vocab)

stoi = {word: i for i, word in enumerate(vocab)}
itos = {i: word for word, i in stoi.items()}

def encode(words):
    return [stoi[word] for word in words]

def decode(indices):
    return ' '.join([itos[i] for i in indices])

data = torch.tensor(encode(words), dtype=torch.long)