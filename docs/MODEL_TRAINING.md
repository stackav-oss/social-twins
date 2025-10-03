# Training on WOMD

## Training a single experiment

To run a training experiment:
```bash
uv run -m scenetokens.train model=[model_name]
```
where `model_name`: either of `wayformer`, `scenetransformer`, `scenetokens_student`, `scenetokens_teacher` or `scenetokens_teacher_unmasked`. The model name needs to be specified.

Additional command line arguments:
* `logger`: either of `mlflow`, `neptune`, `tensorboard`, `wandb`, `csv` or `many_loggers` (which will use both `mlflow` and `csv`). Specific parameters might need to be set for some loggers. **Default** value is `many_loggers`.
* `scenario`: either of `waymo` or `nuscenes`. This will simply set the scenario sequence partition. **Default** value is `waymo`, which will partition the scenario into 1.1 seconds of history and 8 seconds for prediction.
* `paths`: either of `waymo ` (Experiment 1), `waymo_causal_labeled` (Experiment 2), `waymo_causal_unlabeled` (Experiment 3), `safeshift` (Experiment 4), or `safeshift_causal` (Experiment 5). Each specifies the paths to the train/val/test data. **Default** value is `waymo`. See this [doc](./EXPERIMENTS.md) for more details on each experiment.
* `trainer`: either of `cpu`, `ddp`, `gpu` or `mps`. **Default** value is `gpu'.
* `dataset`: This specifies the input data representation. Currently, the only supported value is `waymo`. See this [doc](./DATA_PREPARATION.md) for more details on how to prepare the data.


## Logging Details

#### CSV (Default)
Outputs will be saved to `out/logs/runs/date/experiment_name/csv`.

####  MLflow (Default)
Currently, it needs **tracking_uri** specification, as:
```bash
uv run -m scenetokens.train model=wayformer logger.mlflow.tracking_uri=[uri]
```

#### Tensorboard
To visualize logs:
```bash
uv run tensorboard --logdir out/ --host [host-address] --port [port]
```

#### Other
The other loggers (`neptune`, `wandb`) have not been configured yet, but have pytorch-lightning support. See this [link](https://lightning.ai/docs/pytorch/stable/api_references.html#loggers) for reference.

## Evaluating a single experiment
To run an evaluation, specify any additional config arguments as above and a checkpoint name.
```bash
uv run -m scenetokens.eval ckpt_path=/path/to/the/ckpt.pth model=[model_name]
```

## Debugging
There are various debugging configurations which can be enabled by adding `debug=[debug_name]` to the command, where `debug_name` is either of:
* `default`: runs one epoch on debug mode on cpu.
* `fdr`: runs 1 train, 1 validation and 1 testing step.
* `limit`: runs n epochs with 1% of the training data dn 5% of the val/test data.
* `overfit`: runs n epochs to overfit on b batches.
* `profiler`: runs a performance profiler experiment.

Example, running the profiler:
```bash
uv run -m scenetokens.train model=wayformer debug=profiler
```

# Multirun training (Parameter Sweeps)

To run a sweep of experiments use `-m` and specify in the command line the parameter(s) to be sweeped. For example:
```bash
uv run -m scenetokens.train -m model=[model_name] model.config.num_classes=10,20,50,100
```
This will launch 4 sequential experiments where the value `num_classess` will be set to 10, 20, 50 and 100, respectively. The experiment logs will be saved to `out/logs/multiruns` instead of `out/logs/runs/`.
