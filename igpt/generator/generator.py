import os

import torch

from igpt.model.igpt_model import GPT
from igpt.utils.misc import get_device, setup_logging, visualize_image


class Generator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.img_size = self.cfg.DATA.img_size
        self.seq_len = self.img_size**2
        self.state_dict_path = os.path.join(self.cfg.GENERATOR.model_dir, f"igpt_model_{self.cfg.DATA.dataset}.pth")
        self.logger, self.writer, self.experiment_dir = setup_logging(self.cfg)
        self.build_model()

    def build_model(self):
        self.device = get_device(self.logger)
        self.model = GPT(self.cfg)
        state_dict = torch.load(self.state_dict_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)

    def generate(self):
        self.model.eval()
        with torch.no_grad():
            context = torch.full((self.cfg.GENERATOR.batch_size, 1), 256, dtype=torch.long, device=self.device)
            out = self.model.generate(context, self.seq_len)
            visualize_image(self.cfg, out, self.img_size)
