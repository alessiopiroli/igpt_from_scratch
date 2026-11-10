import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from igpt.utils.misc import get_device, log_data, log_model, setup_logging, log_epoch
from igpt.dataset.fashion_mnist_dataset import FashionMNISTDataset
from igpt.dataset.emnist_dataset import EMNISTDataset
from igpt.model.igpt_model import GPT
from tqdm import tqdm


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_epochs = self.cfg.TRAINING.n_epochs
        self.batch_size = self.cfg.TRAINING.batch_size
        self.lr = float(self.cfg.TRAINING.lr)
        self.logger, self.writer, self.experiment_dir = setup_logging(self.cfg)
        self.build_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.build_dataloaders()
        self.training_steps = 1
        self.validation_steps = 1


    def build_model(self):
        self.device = get_device(self.logger)
        self.model = GPT(self.cfg)
        self.model = self.model.to(self.device)
        log_model(self.cfg, self.logger, self.model, self.batch_size, self.device)


    def build_dataloaders(self):
        if self.cfg.DATA.dataset == "fmnist":
            self.train_dataset = FashionMNISTDataset(self.cfg, train=True, download=True)
            self.validation_dataset = FashionMNISTDataset(self.cfg, train=False, download=True)
        elif self.cfg.DATA.dataset == "emnist":
            self.train_dataset = EMNISTDataset(self.cfg, train=True, download=True)
            self.validation_dataset = EMNISTDataset(self.cfg, train=False, download=True)
        else:
            raise ValueError("Dataset not recognized!")

        self.train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.validation_dataset, self.batch_size, shuffle=False)

        log_data(self.logger, self.train_dataset, self.validation_dataset, self.train_loader, self.validation_loader)


    def train(self):
        for epoch in range(self.n_epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch(epoch)

            log_epoch(
                self.logger,
                self.writer,
                epoch,
                self.n_epochs,
                train_loss,
                val_loss,
                self.experiment_dir,
                self.model
            )


    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Training epoch {epoch+1}")


        for model_input, target in pbar:
            self.optimizer.zero_grad()
            model_input, target = model_input.to(self.device), target.to(self.device)
            prediction = self.model(model_input)
            prediction = torch.flatten(prediction, 0, 1)
            target = torch.flatten(target, 0, -1)
            loss = self.loss_fn(prediction, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            self.writer.add_scalar("Training-Loss-Step", loss.item(), self.training_steps)
            self.training_steps += 1
            
        return total_loss / len(self.train_loader)


    def validate_one_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        pbar = tqdm(self.validation_loader, f"Validating epoch {epoch+1}")

        with torch.no_grad():
            for model_input, target in pbar:
                model_input, target = model_input.to(self.device), target.to(self.device)
                prediction = self.model(model_input)
                prediction = torch.flatten(prediction, 0, 1)
                target = torch.flatten(target, 0, -1)
                loss = self.loss_fn(prediction, target)
                total_loss += loss.item()
                self.writer.add_scalar("Validation-Loss-Step", loss.item(), self.validation_steps)
                self.validation_steps += 1

        return total_loss / len(self.validation_loader)