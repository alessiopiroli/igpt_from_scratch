import torch

from igpt.utils.misc import get_device, setup_logging, visualize_image
from igpt.model.igpt_model import GPT


class Generator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.img_size = self.cfg.DATA.img_size
        self.seq_len = self.img_size ** 2
        self.state_dict_path = self.cfg.GENERATOR.state_dict_path
        self.logger, self.writer, self.experiment_dir = setup_logging(self.cfg)
        self.build_model()

    def build_model(self):
        self.device = get_device(self.logger)
        self.model = GPT(self.cfg)
        self.model.load_state_dict(torch.load(self.state_dict_path))
        self.model = self.model.to(self.device)

    def generate(self):
        self.model.eval()
        context = torch.full((1, 1), 256, dtype=torch.long, device=self.device)
        out = self.model.generate(context, self.seq_len)
        visualize_image(out, self.img_size)
