from typing import Any, Dict, List, Tuple
from models_mae import mae_vit_large_patch16
from pprint import pprint
import torch

model = mae_vit_large_patch16()

checkpoint = torch.load('pretrained_weights/mae_pretrain_vit_large.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)

keys = list(model.state_dict().keys())

selected_keys: List[Tuple[str, str]] = []
# selected_state_dict: Dict[str, Any] = {}
# model.state_dict()

for key in keys:
    if key.startswith('blocks.'):
        selected_keys.append((key, key.replace('blocks.', 'encoder.')))
    if key == 'cls_token':
        selected_keys.append((key, 'global_tokens'))
    if key.startswith('patch_embed.'):
        selected_keys.append((key, key.replace('patch_embed.', 'input_adapters.rgb.')))
        
for (key, mapping_key) in selected_keys:
    torch.save()

with open('key.txt', 'w') as f:
    f.write('\n'.join(keys))
    
pprint(selected_keys)

# print(model)