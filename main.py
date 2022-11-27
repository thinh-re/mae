from models_mae import mae_vit_large_patch16
import torch

model = mae_vit_large_patch16()

checkpoint = torch.load('pretrained_weights/mae_pretrain_vit_large.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
