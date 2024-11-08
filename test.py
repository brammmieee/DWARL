from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(config_path='config', config_name='test', version_base='1.1')
def main(cfg: DictConfig):
    
    print(cfg)

if __name__ == "__main__":
    main()