# run_train.py

from omegaconf import OmegaConf
from state.config import get_cfg  # or wherever config is constructed
from state.tx.train import train  # adjust import path to exact function

def main():
    # 1. Load base config (e.g., default TOML or YAML)
    cfg = get_cfg("examples/fewshot.toml")
    
    # 2. Override parameters (like batch size, learning rate)
    overrides = {
        "training.batch_size": 64,
        "training.lr": 1e-4,
        "data.kwargs.embed_key": "X_hvg",
        # Add more overrides here
    }
    cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    
    # 3. Create output directory, logging/W&B, etc. if needed

    # 4. Run training
    train(cfg)

if __name__ == "__main__":
    main()
