import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config: DictConfig):
    if "train" in config.exp.run_func:
        from experiments.train_pointnet import train_purity
        train_purity(config)
    elif "explain" in config.exp.run_func:
        from experiments.explain_pointnet import explain_predictions
        explain_predictions(config)
    else:
        raise ValueError(f"Unknown run_func for PointNet: {config.exp.run_func}")


if __name__ == "__main__":
    main()
