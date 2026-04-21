# Using multimodal satellite data with vision transformer for object tracking (ship tracking)

The vision transformer architecture is used to enable wide area continuous ship tracking with optical and synthetic aperture radar (SAR) data. Additionally, a diffusion model is leveraged for cross-modal sample generation and feature fusion with original features (sensor fusion) during inference in order to enhance discriminability.

## Installation


### Setup
```bash
conda create -n mos python=3.11
conda activate mos
pip install -r requirements.txt
```

### Train Model

```
python train.py --config_file configs/hoss_transoss.yml
```


### Evaluate Model

```
python test.py --config_file configs/hoss_transoss.yml TEST.WEIGHT 'the checkpoint path'
```

### Utilize BBDM to generate SAR samples

Please refer to [BBDM](https://arxiv.org/pdf/2205.07680) for training the diffusion model. The BBDM is trained for 100 epochs on [QXS-SAROPT dataset](https://github.com/yaoxu008/QXS-SAROPT) and fine-tuned for 250 epochs on the HOSS ReID training set.

After training the optical-SAR diffusion model, the optical images in HOSS ReID test set can be transferred to SAR modality. The generated image dir should be specified in the `self.queryAdd_dir` and `self.galleryAdd_dir` variables within `/datasets/hoss.py`.
