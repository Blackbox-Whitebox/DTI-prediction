import logging
import utils
from tqdm import tqdm
import torch

class RunningAverager:
    def __init__(self, track=[], smooth=.6):
        self.track = {
            to_track: 0
            for to_track in track
        }
        self.smooth = smooth

    def add_new(self, values):
        for key in values.keys():
            if key not in self.track:
                logging.warning(f"{key} not in tracked values.")
            
            new_value = values[key]
            old_value = self.track[key]

            self.track[key] = self.smooth*new_value + (1-self.smooth)*old_value

    def get_tracked(self):
        return self.track


class Trainer:
    def __init__(
        self,
        model,
        batch_size,
        train,
        test,
        epochs,
        optimizer,
        lossfn,
        metrics,
        writer,
        config_file,
        smooth=.6,
        device="cuda",
    ):
        self.model = model
        self.model.to(device)
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lossfn = lossfn
        self.metrics = metrics
        self.train = train
        self.test = test
        self.epochs = epochs
        self.writer = writer
        self.smooth = smooth
        self.device = device
        self.config_file = config_file

        self.lowest_loss = {
            "train": float("inf"),
            "test": float("inf")
        }

        self.highest_accuracy = {
            "test": float("-inf")
        }

    def train_one_epoch(self, epoch):
        averager = RunningAverager(
            track=["loss"]+list(self.metrics.keys()),
            smooth=self.smooth
        )

        for i, (x1, x2, y) in enumerate(self.train):
            # print(x1.shape, x2.shape)
            self.model.train()

            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            y = y.to(self.device)

            p = self.model((x1, x2))
            loss = self.lossfn(p, y)

            loss.backward()

            # if (i+1) % self.batch_size == 0 or (i+1) == len(self.train):
                # self.optimizer.step()
                # self.optimizer.zero_grad()

            self.optimizer.step()
            self.optimizer.zero_grad()

            averager.add_new({
                "loss": loss.item(),
            })

            for metric_name in self.metrics.keys():
                metric_value = self.metrics[metric_name](p, y)
                
                averager.add_new({
                    metric_name: metric_value
                })

        averaged = averager.get_tracked()

        utils.write_to_tb(
            t=epoch,
            writer=self.writer,
            scalars={
                f"{metric_name}/Train": averaged[metric_name] for metric_name in list(averaged.keys())
            },
            net=self.model
        )

        return averaged

    @torch.no_grad()
    def test_one_epoch(self, epoch):
        self.model.eval()
        averager = RunningAverager(
            track=["loss"]+list(self.metrics.keys()),
            smooth=self.smooth
        )

        for x1, x2, y in self.test:

            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            y = y.to(self.device)
            
            p = self.model((x1, x2))
            loss = self.lossfn(p, y)

            averager.add_new({
                "loss": loss.item(),
            })

            for metric_name in self.metrics.keys():
                metric_value = self.metrics[metric_name](p, y)
                
                averager.add_new({
                    metric_name: metric_value
                })

        averaged = averager.get_tracked()

        utils.write_to_tb(
            t=epoch,
            writer=self.writer,
            scalars={
                f"{metric_name}/Test": averaged[metric_name] for metric_name in list(averaged.keys())
            },
            net=self.model
        )

        return averaged

    def run(self):
        for epoch in tqdm(range(self.epochs)):
            train_averaged = self.train_one_epoch(epoch)
            
            test_averaged = self.test_one_epoch(epoch)

            if test_averaged["loss"] < self.lowest_loss["test"] or test_averaged["Accuracy"] > self.highest_accuracy["test"]:
                self.lowest_loss["test"] = test_averaged["loss"]
                self.highest_accuracy["test"] = test_averaged["Accuracy"]
                
                self.save_checkpoint(accuracy=test_averaged["Accuracy"], epoch=epoch)

            if train_averaged["loss"] < self.lowest_loss["train"]:
                self.lowest_loss["train"] = train_averaged["loss"]


    def save_checkpoint(self, accuracy=None, epoch=None):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss": self.lowest_loss["test"],
            "accuracy": accuracy,
            "epoch": epoch+1
        }, f"../checkpoints/classification/{self.config_file['RUN_NAME']}/model.pt")
