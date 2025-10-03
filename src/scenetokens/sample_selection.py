"""Evaluation script for SocialTokens.

Example usage:

    uv run -m scenetokens.sample_selection \
        +model.config.sample_selection=true
        cache=true \
        paths=[path_config] model=[model_config] sweep_tag=[sweep_tag] ckpt_name=[epoch_x]
"""

import random
from pathlib import Path

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers.logger import Logger
from torch.utils.data import DataLoader, Dataset

from scenetokens import analysis, utils
from scenetokens.schemas import output_schemas as output
from scenetokens.utils.constants import DataSplits


log = utils.get_pylogger(__name__)

torch.set_float32_matmul_precision("medium")
root_path = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

utils.disable_mlflow_tls_verification()


@utils.task_wrapper
def evaluate_and_cache_dataset(cfg: DictConfig) -> tuple[dict, dict]:
    """Evaluates and caches the training set for sample selection purposes.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    if not Path(cfg.ckpt_path).exists():
        error_message = f"Checkpoint path: {cfg.ckpt_path} does not exist!"
        raise ValueError(error_message)

    log.info("Instantiating model <%s>", cfg.model._target_)
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: list[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating trainer <%s>", cfg.trainer._target_)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {"cfg": cfg, "model": model, "logger": logger, "trainer": trainer}
    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Instantiating testing dataset <%s>", cfg.dataset._target_)
    cfg.dataset.config.split = DataSplits.TRAINING
    training_set: Dataset = hydra.utils.instantiate(cfg.dataset)
    test_loader = DataLoader(
        training_set,
        batch_size=cfg.model.config.eval_batch_size,
        num_workers=cfg.model.config.load_num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=training_set.collate_fn,
    )
    log.info("Starting test process to cache training set.")
    trainer.test(model=model, dataloaders=test_loader, ckpt_path=cfg.ckpt_path)
    metric_dict = trainer.callback_metrics
    return metric_dict, object_dict


def run_analysis(cfg: DictConfig, batches: dict[str, output.ModelOutput]) -> None:
    """Runs distribution analysis on training set.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        batches (dict[str, ModelOutput]): Dictionary containing the model outputs of the training set.
    """
    random.seed(cfg.seed)

    output_path = Path(f"{cfg.output_path}/{cfg.split}_model-analysis")
    output_path.mkdir(parents=True, exist_ok=True)

    # Produces a histogram over the tokenized scenarios, representing token-utilization.
    log.info("Running tokenization distribution analysis...")
    analysis.plot_scenario_class_distribution(cfg, batches, output_path)

    # Produces a visualization of the scenario embeddings using dimensionality reduction methods.
    log.info("Running dimensionality reduction analysis...")
    dim_reduction_result = analysis.compute_dimensionality_reduction(cfg, batches, output_path)
    analysis.plot_manifold_by_tokens(cfg, dim_reduction_result, batches, output_path)


@hydra.main(version_base="1.3", config_path="configs", config_name="sample_selection.yaml")
def main(cfg: DictConfig) -> None:
    """Hydra's entrypoint for running the model's evaluation."""
    # Apply extra utilities (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # Run model evaluation to cache the training set embeddings
    if cfg.cache:
        evaluate_and_cache_dataset(cfg)

    batches = utils.load_batches(cfg.batches_path, cfg.num_batches, cfg.num_scenarios, cfg.seed, cfg.split)

    # Run distribution analysis over the training set
    if cfg.run_analysis:
        run_analysis(cfg, batches)

    # Run sample selection
    output_path = root_path / "meta"
    output_path.mkdir(parents=True, exist_ok=True)
    analysis.run_sample_selection(cfg, batches, output_path)


if __name__ == "__main__":
    main()
