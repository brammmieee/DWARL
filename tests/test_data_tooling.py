from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import data.data_generator as dg
import data.data_set as ds
import hydra

@hydra.main(config_path='../config', config_name='train', version_base=str(1.1))
def main(cfg: DictConfig):
    print("Data configuration:")
    print(OmegaConf.to_yaml(cfg.data))
    
    # Erase then generate data
    data_generator = dg.DataGenerator(cfg.data, cfg.paths)
    data_generator.erase_data(); data_generator.generate_data()
    
    # # Create dataset
    # dataset = ds.Dataset(cfg.data, cfg.paths)

    # # Create the DataLoader with desired batch size and shuffle setting
    # data_loader = DataLoader(
    #     dataset,
    #     batch_size=32, 
    #     shuffle=True
    # )
    
    # # import ipdb; ipdb.set_trace()

    # # Use the DataLoader within your environment
    # for batch in data_loader:
    #     grids = batch["grid"]
    #     protos = batch["proto"]
    #     init_poses = batch["init_pose"]
    #     goal_poses = batch["goal_pose"]
        
    #     print(f"\n")
    #     print("Grids (sample):", grids[:2])  # Print only the first 2 grids
    #     print("Protos (sample):", protos[:2])  # Print only the first 2 protos
    #     print("Initial Poses:", init_poses)
    #     print("Goal Poses:", goal_poses)


if __name__ == '__main__':
    main()