from typing import Any, Dict, List, Tuple, OrderedDict
from models_mae import mae_vit_large_patch16
from pprint import pprint
import torch
from torch import Tensor

model = mae_vit_large_patch16()

checkpoint = torch.load('pretrained_weights/mae_pretrain_vit_large.pth', map_location='cpu')
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
        
torch.save(OrderedDict(model=selected_state_dict), 'pretrained_weights/selected_mae_pretrain_vit_large.pth')

with open('key.txt', 'w') as f:
    f.write('\n'.join(keys))
