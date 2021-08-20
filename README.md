# OverLORD - Official PyTorch Implementation 
> [Scaling-up Disentanglement for Image Translation](http://www.vision.huji.ac.il/overlord)  
> Aviv Gabbay and Yedid Hoshen  
> International Conference on Computer Vision (ICCV), 2021.

> **Abstract:** Image translation methods typically aim to manipulate a set of labeled attributes (given as supervision at training time e.g. domain label) while leaving the unlabeled attributes intact. Current methods achieve either: (i) disentanglement, which exhibits low visual fidelity and can only be satisfied where the attributes are perfectly uncorrelated. (ii) visually-plausible translations, which are clearly not disentangled. In this work, we propose OverLORD, a single framework for disentangling labeled and unlabeled attributes as well as synthesizing high-fidelity images, which is composed of two stages; (i) Disentanglement: Learning disentangled representations with latent optimization. Differently from previous approaches, we do not rely on adversarial training or any architectural biases. (ii) Synthesis: Training feed-forward encoders for inferring the learned attributes and tuning the generator in an adversarial manner to increase the perceptual quality. When the labeled and unlabeled attributes are correlated, we model an additional representation that accounts for the correlated attributes and improves disentanglement. We highlight that our flexible framework covers multiple settings as disentangling labeled attributes, pose and appearance, localized concepts, and shape and texture. We present significantly better disentanglement with higher translation quality and greater output diversity than state-of-the-art methods.

<a href="https://arxiv.org/abs/2103.14017" target="_blank"><img src="https://img.shields.io/badge/arXiv-2103.14017-b31b1b.svg"></a>

## Description

### Case 1: Uncorrelated Labeled and Unlabeled Attributes
- Facial age editing (FFHQ)

| Input | [0-9] | [10-19] | [20-29] | [30-39] | [40-49] | [50-59] | [60-69] | [70-79] |
|:-----:|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|   ![](http://www.vision.huji.ac.il/overlord/img/aging/a/0-0.jpg)   |   ![](http://www.vision.huji.ac.il/overlord/img/aging/a/0-1.jpg)   |    ![](http://www.vision.huji.ac.il/overlord/img/aging/a/0-2.jpg)    |    ![](http://www.vision.huji.ac.il/overlord/img/aging/a/0-3.jpg)    |    ![](http://www.vision.huji.ac.il/overlord/img/aging/a/0-4.jpg)    |    ![](http://www.vision.huji.ac.il/overlord/img/aging/a/0-5.jpg)    |    ![](http://www.vision.huji.ac.il/overlord/img/aging/a/0-6.jpg)    |    ![](http://www.vision.huji.ac.il/overlord/img/aging/a/0-7.jpg)    |    ![](http://www.vision.huji.ac.il/overlord/img/aging/a/0-8.jpg)    |
|   ![](http://www.vision.huji.ac.il/overlord/img/aging/b/0-0.jpg)   |   ![](http://www.vision.huji.ac.il/overlord/img/aging/b/0-1.jpg)   |    ![](http://www.vision.huji.ac.il/overlord/img/aging/b/0-2.jpg)    |    ![](http://www.vision.huji.ac.il/overlord/img/aging/b/0-3.jpg)    |    ![](http://www.vision.huji.ac.il/overlord/img/aging/b/0-4.jpg)    |    ![](http://www.vision.huji.ac.il/overlord/img/aging/b/0-5.jpg)    |    ![](http://www.vision.huji.ac.il/overlord/img/aging/b/0-6.jpg)    |    ![](http://www.vision.huji.ac.il/overlord/img/aging/b/0-7.jpg)    |    ![](http://www.vision.huji.ac.il/overlord/img/aging/b/0-8.jpg)    |


- Identity (CelebA)


## Getting Started

## Pretrained Models

## Training

## Inference

## Citation
```
@inproceedings{gabbay2021overlord,
  author    = {Aviv Gabbay and Yedid Hoshen},
  title     = {Scaling-up Disentanglement for Image Translation},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year      = {2021}
}
```
