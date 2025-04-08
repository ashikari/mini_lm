from data import get_train_dataloader
from argparse import ArgumentParser, Namespace
from torch.optim import Adam

class Trainer:
    def __init__(self, batch_size: int, limit: int | None = None):
        self.batch_size = batch_size
        self.limit = limit
        self.train_dataloader = get_train_dataloader(self.batch_size)

        # add optimizer
        self.model = None # TODO: add model
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

    def train(self):
        for i, element in enumerate(self.train_dataloader):
            # TODO: add model inference
            # TODO: add loss calculation
            # loss = None

            # update model parameters
            self.optimizer.zero_grad()
            # loss.backward()
            self.optimizer.step()

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
