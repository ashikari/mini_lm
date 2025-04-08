from data import get_train_dataloader
from argparse import ArgumentParser, Namespace


class Trainer:
    def __init__(self, batch_size: int, limit: int | None = None):
        self.batch_size = batch_size
        self.limit = limit
        self.train_dataloader = get_train_dataloader(self.batch_size)

    def train(self):
        for i, element in enumerate(self.train_dataloader):
            print(element.keys())
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
