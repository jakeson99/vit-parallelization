import pydra

from vit_scratch.train import run_training
from vit_scratch.configs.configs import ExperimentConfig
import mlflow


@pydra.main(ExperimentConfig)
def main(cfg: ExperimentConfig):
    # Start an mlflow experiment
    mlflow.set_experiment("ViT from Scratch Single Node Experiment")

    # Enable system metrics monitoring
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)
    run_training(cfg)


if __name__ == "__main__":
    main()
