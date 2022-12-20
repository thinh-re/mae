from typing import Dict, Union

import wandb
from wandb.sdk.wandb_run import Run

WANDB_PROJECT_NAME = 'MAE'
WANDB_API_KEY = 'c3fc6b778d58b02a1519dec88b08f0dae1fd5b3b'

def wandb_login() -> None:
    wandb.login(key=WANDB_API_KEY)

def wandb_init(
    name: str,
    config: Union[Dict, str, None] = None
) -> Run:
    return wandb.init(
        name=name, project=WANDB_PROJECT_NAME, 
        id=name, resume='auto', config=config
    )
