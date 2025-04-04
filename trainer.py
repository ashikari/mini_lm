from datasets import load_dataset

from torch.utils.data import DataLoader
import torch

from transformers import AutoTokenizer

from argparse import ArgumentParser, Namespace

# Initialize the BERT tokenizer
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


class Trainer:
    def __init__(self, batch_size: int, limit: int | None = None):
        self.batch_size = batch_size
        self.limit = limit
        self.dataset = load_dataset(
            "cambridge-climb/BabyLM", trust_remote_code=True, split="train[:1%]"
        )

        # Tokenize the datasets
        self.tokenized_datasets = self.dataset.map(tokenize, batched=False)
        self.train_dataloader = DataLoader(
            self.tokenized_datasets,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def train(self):
        for i, element in enumerate(self.train_dataloader):
            print(element.keys())
            # print(i, element["text"])
            print("Input IDs shape")
            print(element["input_ids"].shape)
            print(i, len(element["input_ids"]))
            print(i, len(element["input_ids"][0]))

            if self.limit and i == self.limit:
                break

    def evaluate(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass


def get_args() -> Namespace:
    parser = ArgumentParser(description="Train a mini language model")
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for training"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit the number of training batches"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Setup argument parser
    args = get_args()

    trainer = Trainer(batch_size=args.batch_size, limit=args.limit)
    trainer.train()
