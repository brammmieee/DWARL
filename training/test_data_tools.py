from omegaconf import DictConfig, OmegaConf
import hydra
import yaml
from pathlib import Path
import utils.data_generator as dg

@hydra.main(config_path='../config', config_name='train', version_base=str(1.1))
def main(cfg: DictConfig):

    ### TESING DATA GENERATOR ###
    print("Data generator Config:")
    print(OmegaConf.to_yaml(cfg.data_generator))
    
    # Erase then generate data
    data_generator = dg.DataGenerator(cfg.data_generator, cfg.paths)
    data_generator.erase_data(); data_generator.generate_data()
    
    # import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()