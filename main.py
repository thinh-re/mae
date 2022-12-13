from typing import OrderedDict
from models_mae import mae_vit_large_patch16, mae_vit_huge_patch14
from pprint import pprint
import torch
from torch import Tensor

MODEL_TYPE = 'huge'

if MODEL_TYPE == 'large':
    model = mae_vit_large_patch16()
elif MODEL_TYPE == 'huge':
    model = mae_vit_huge_patch14()

checkpoint = torch.load(f'pretrained_weights/mae_pretrain_vit_{MODEL_TYPE}.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)

keys = list(model.state_dict().keys())

state_dict: OrderedDict[str, Tensor] = model.state_dict()
selected_state_dict = OrderedDict()
for key, value in state_dict.items():
    if key.startswith('blocks.'):
        selected_state_dict[key.replace('blocks.', 'encoder.')] = value
    if key == 'cls_token':
        selected_state_dict['global_tokens'] = value
    if key.startswith('patch_embed.'):
        selected_state_dict[key.replace('patch_embed.', 'input_adapters.rgb.')] = value
        
torch.save(OrderedDict(model=selected_state_dict), f'pretrained_weights/selected_mae_pretrain_vit_{MODEL_TYPE}.pth')

with open('key.txt', 'w') as f:
    f.write('\n'.join(keys))
