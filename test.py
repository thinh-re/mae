from typing import Any, Dict, List, Tuple, OrderedDict
from models_mae import mae_vit_large_patch16
from pprint import pprint
import torch
from torch import Tensor
import json

def save_key(keys: List[str], file_path: str) -> None:
    json_object = json.dumps(
        dict(keys=keys), 
        indent=4, 
        ensure_ascii=False
    ).encode('utf8')

    with open(file_path, "wb") as f:
        f.write(json_object)

model = mae_vit_large_patch16()

checkpoint = torch.load('pretrained_weights/selected_mae_pretrain_vit_large.pth', map_location='cpu')
keys = list(checkpoint['model'].keys())
save_key(keys, 'selected_large_key.json')

checkpoint = torch.load('pretrained_weights/mae_pretrain_vit_large.pth', map_location='cpu')
keys = list(checkpoint['model'].keys())
save_key(keys, 'mae_large_key.json')
