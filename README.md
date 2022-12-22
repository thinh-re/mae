# Masked Autoencoders (MAE): Train on Kaggle

This is an example of training MAE on Kaggle

## Environments

- Kaggle: enabled GPU T4x2
- Python: 3.10
- Datasets: [mae-v1](https://www.kaggle.com/datasets/thinhhuynh3108/maev1) (8,025 images) or [ImageNet1K](https://www.kaggle.com/competitions/imagenet-object-localization-challenge) (1,281,167 images)
- Log to `wandb.ai`

## How to run

- Import ipynb `mae-kaggle.ipynb` into Kaggle.
- Choose one of these datasets: [mae-v1](https://www.kaggle.com/datasets/thinhhuynh3108/maev1) (8,025 images) or [ImageNet1K](https://www.kaggle.com/competitions/imagenet-object-localization-challenge) (1,281,167 images)
- Use your own Wandb API token
- Update hyperparameters or leave them as default
- Run all and enjoy!

## Key Takeaways

- I do not guarantee to reproduce the same results as in the paper (it takes too much time to train the entire ImageNet1K)
- I notice that we can set the batch size up to 130 per GPU (that means 260 in total) when training the base model (mae_vit_base_patch16)

## Acknowledgement

This repository is inspired by [MAE](https://github.com/facebookresearch/mae) (Original paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) )

```bash
@inproceedings{he2022masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16000--16009},
  year={2022}
}
```
