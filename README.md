# ImageGPT from scratch
<!-- <div style="display: flex; gap: 5px;">
<a href="https://arxiv.org/abs/2105.15203">
<img src="https://img.shields.io/badge/Arxiv-Paper-green" alt="Arxiv Paper" />
</a>
</div> -->
<div style="display: flex; gap: 5px;">
  <a href="https://proceedings.mlr.press/v119/chen20s.html">
    <img src="https://img.shields.io/badge/PMLR-Paper-green" alt="Generative Pretraining from Pixels" />
  </a>
</div>



###
Implementing ["Generative Pretraining from Pixels"](https://proceedings.mlr.press/v119/chen20s.html) from scratch.

### Clone and install dependencies
``` 
git clone https://github.com/alessiopiroli/igpt_from_scratch
pip install -r requirements.txt 
```

### Train 
``` 
python train.py igpt/config/igpt_config.yaml
```


### Generate
``` 
python generate.py igpt/config/igpt_config.yaml
```

### Generation Results
> Sample generations from the model trained on the [FashionMNIST](https://arxiv.org/abs/1708.07747) dataset.

<p align="center">
  <img src="artifacts/media/igpt.webp" alt="Sample generations from the model trained on the FashionMNIST dataset.">
</p>