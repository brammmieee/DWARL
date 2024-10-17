#!/usr/bin/python3

# %%

from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import hydra

# %%
@hydra.main(config_path='../config', config_name='train', version_base=str(1.1))
def main(cfg: DictConfig):

   # TESTING HY#DRA
    print(OmegaConf.to_yaml(cfg))
    # print(f"Hello, {cfg.train.steps}!")
    # cfg.train.steps = 2
    # print(Path(cfg.file_paths.root))
    import ipdb; ipdb.set_trace()

    import os
    print(f"Current working directory: {os.getcwd()}")


if __name__ == '__main__':
    main()
 

# %%