# ESG-Net: Event-Aware Semantic Guided Network for Dense Audio-Visual Event Localization
[[Early Access]](https://ieeexplore.ieee.org/document/11417281) [[ArXiv]](https://arxiv.org/abs/2507.09945)

This repository contains the code for our TMM paper "ESG-Net: Event-Aware Semantic Guided Network for Dense Audio-Visual Event Localization".

# Data Preparation
<!-- #### Download features and annotations -->
Please follow the repo. of [UnAV](https://github.com/ttgeng233/UnAV) and [UniAV](https://github.com/ttgeng233/UniAV) to download the audio/visual features and annotations of UnAV-100 dataset. 

Note that the features of the former are extracted by I3D+VGGish+RAFT, while the features of the latter are extracted by ONE‑PEACE.

After downloading the dataset, unpack the files under `./data`. The folder structure should look like this:
```
This folder
│   README.md
│   ...  
└───data/
│    └───unav100/
│    	 └───annotations
│    	 └───av_features  
└───libs
│   ...
```

# Training
Run `train.py` to train ESG-Net on UnAV-100.
```
python ./train.py ./configs/avel_unav100.yaml --output reproduce
```

# Evaluation
Run ```eval.py``` to evaluate the trained model.
```
python ./eval.py ./configs/avel_unav100.yaml ./ckpt/avel_unav100_reproduce
```

# Citation
Coming soon.
Our paper is currently in the Early Access stage. 

# Acknowledgement
Our codes are developed based on the codebase of pioneering work UnAV. We thank the authors for their excellent efforts.
