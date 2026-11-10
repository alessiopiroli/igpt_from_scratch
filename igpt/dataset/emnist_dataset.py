import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import EMNIST


class EMNISTDataset(Dataset):
    def __init__(
        self,
        cfg,
        train=True,
        download=True,
        split="balanced"
    ):
        self.cfg = cfg
        self.img_size = self.cfg.DATA.img_size
        self.seq_len = self.img_size ** 2
        self.boi_token = torch.tensor(
            [self.cfg.DATA.boi_token]
        ).to(torch.long)

        self.dataset = EMNIST(
            root=self.cfg.DATA.root_dir,
            split=split,
            train=train,
            download=download,
            transform=transforms.PILToTensor()
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        target, _ = self.dataset[idx]

        target = torch.flatten(target, 0, -1).to(torch.long)
        model_input = torch.cat((self.boi_token, target[:-1]), dim=0)

        return model_input, target
