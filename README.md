# GPT 0.1

A simple text generation project using a word-level RNN built with PyTorch.

## 📄 Overview

GPT 0.1 is a minimalistic text generator that trains on a text corpus and generates sequences of words based on a starting prompt. It uses a custom RNN model to predict the next word step by step, showcasing fundamental NLP techniques.

## 🗂️ Project Structure

```

.
├── **pycache**/
├── config.py          # Configuration settings
├── dataset.py         # Dataset loading and tokenization
├── generate.py        # Script to generate text using the trained model
├── input.txt          # Training text corpus
├── model.py           # RNN model definition
├── train.py           # Model training script
├── word\_model.pth     # Saved model weights
└── requirements.txt   # Python dependencies

````

## 🚀 How to Use

1️⃣ **Install dependencies:**

```bash
pip install -r requirements.txt
````

2️⃣ **Train the model**
- Insert your text (can be any text, even your message history with a friend) into input.txt
- Launch train.py


3️⃣ **Generate text:**

Set your desired starting word or sentence on line 26 of generate.py and launch generate.py to begin text generation.

Note that starting sentence or a word should be included into input.txt, else the text won't be generated

## 📦 Requirements
pip install -r requirements.txt

## ✍️ Author - Swino4ka
