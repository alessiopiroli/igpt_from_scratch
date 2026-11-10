import pytest
import torch

from igpt.model.igpt_model import GPT, AttnHead, DecoderBlock, FeedForward, MultiHeadAttention
from igpt.utils.misc import load_config


@pytest.fixture
def cfg():
    return load_config("igpt/config/igpt_config.yaml")


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_igpt(cfg, bs):
    img_size = cfg.DATA.img_size
    seq_len = img_size**2

    x = torch.randint(0, 255, (bs, seq_len))
    gpt = GPT(cfg)
    out = gpt(x)

    assert out.shape == (bs, seq_len, cfg.MODEL.vocab_size - 1)


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_decoder_block(cfg, bs):
    img_size = cfg.DATA.img_size
    seq_len = img_size**2
    embed_dim = cfg.MODEL.embed_dim

    x = torch.randn(bs, seq_len, embed_dim)
    decoder_block = DecoderBlock(cfg)
    out = decoder_block(x)

    assert out.shape == x.shape


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_multihead_attention(cfg, bs):
    img_size = cfg.DATA.img_size
    seq_len = img_size**2
    embed_dim = cfg.MODEL.embed_dim

    x = torch.randn(bs, seq_len, embed_dim)
    multihead_attention = MultiHeadAttention(cfg)
    out = multihead_attention(x)

    assert out.shape == x.shape


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_attn_head(cfg, bs):
    img_size = cfg.DATA.img_size
    seq_len = img_size**2
    embed_dim = cfg.MODEL.embed_dim
    n_heads = cfg.MODEL.n_heads
    out_dim = embed_dim // n_heads

    x = torch.randn(bs, seq_len, embed_dim)
    attn_head = AttnHead(cfg)
    out = attn_head(x)

    assert out.shape == (bs, seq_len, out_dim)


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_feedforward(cfg, bs):
    img_size = cfg.DATA.img_size
    seq_len = img_size**2
    embed_dim = cfg.MODEL.embed_dim

    x = torch.randn(bs, seq_len, embed_dim)
    feedforward = FeedForward(cfg)
    out = feedforward(x)

    assert out.shape == x.shape
