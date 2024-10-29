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
    
    # load map name list
    with open(Path(cfg.paths.resources.map_name_lists) / f"{cfg.data_generator.map.list}.yaml") as f:
        map_name_list = yaml.load(f, Loader=yaml.BaseLoader)
        # print(f"Map name list: first 5 maps: {map_name_list[:5]}")
    
    # Generate the data
    data_generator = dg.DataGenerator(cfg.data_generator, cfg.paths)
    data_generator.erase_data() # Erase the data folder
    data_generator.generate_data(map_name_list)
    
    # import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()