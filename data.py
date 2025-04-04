from datasets import load_dataset

from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize(examples):
    # keep return tensors as None because Hugging Face's dataset class cannot return tensors
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, return_tensors=None
    )


def collate_fn(batch):
    """
    Custom collate function for DataLoader that processes a batch of examples.

    Args:
        batch: A list of dictionaries, each containing tokenized text data

    Returns:
        A dictionary with batched tensors for input_ids, attention_mask, etc.
    """
    # Convert lists to tensors
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    attention_mask = torch.tensor([item["attention_mask"] for item in batch])

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def get_train_dataloader(
    batch_size: int, collate_fn: callable = collate_fn
) -> DataLoader:
    dataset = load_dataset(
        "cambridge-climb/BabyLM", trust_remote_code=True, split="train[:1%]"
    )
    tokenized_datasets = dataset.map(tokenize, batched=False)
    train_dataloader = DataLoader(
        tokenized_datasets,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return train_dataloader
