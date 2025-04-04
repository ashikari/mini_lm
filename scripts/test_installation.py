import torch

print("PyTorch Version:", torch.__version__)
print("MPS Available:", torch.mps.is_available())
print("CUDA Available:", torch.cuda.is_available())

from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:1%]")
print("Loaded Dataset:", dataset)

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print("Sample tokenization: ", "Hello world", " -> ", tokenizer.encode("Hello world"))
