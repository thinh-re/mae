from typing import List, OrderedDict
from models_mae import mae_vit_large_patch16, mae_vit_huge_patch14
from pprint import pprint
import torch
import json
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
selected_keys = []
for key, value in state_dict.items():
    if key.startswith('blocks.'):
        key = key.replace('blocks.', 'encoder.')
        selected_state_dict[key] = value
        selected_keys.append(key)
    if key == 'cls_token':
        selected_state_dict['global_tokens'] = value
        selected_keys.append('global_tokens')
    if key.startswith('patch_embed.'):
        key = key.replace('patch_embed.', 'input_adapters.rgb.')
        selected_state_dict[key] = value
        selected_keys.append(key)
        
# torch.save(OrderedDict(model=selected_state_dict), f'pretrained_weights/selected_mae_pretrain_vit_{MODEL_TYPE}.pth')

def save_key(keys: List[str], file_path: str) -> None:
    json_object = json.dumps(
        dict(keys=keys), 
        indent=4, 
        ensure_ascii=False
    ).encode('utf8')

    with open(file_path, "wb") as f:
        f.write(json_object)
        
save_key(selected_keys, f'mae_{MODEL_TYPE}_key.json')

with open('key.txt', 'w') as f:
    f.write('\n'.join(keys))
