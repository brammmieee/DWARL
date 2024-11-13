from omegaconf import DictConfig
from pathlib import Path
import hydra
import re
import utils.test_tools as tt
import json
import os
import yaml

# Set path
json_file_path = '/DWARL/outputs/testing/2024-11-13/10-50-50/results/results.json'
json_file_path = Path(json_file_path)
folder = json_file_path.parent.parent

# Load results and config
results = tt.load_eval_results(json_file_path)
with open(Path(folder, '.hydra', 'config.yaml'), 'r') as f:
    cfg = yaml.safe_load(f)
cfg = DictConfig(cfg)

# Always show plots
cfg.plotter.show = True
plotter = tt.ResultPlotter(cfg.plotter)
plotter.plot_results(results, block=True)

