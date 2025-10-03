# Experiments on WOMD

### Experiment 1: Mini WOMD
- Train on mini-training dataset
- Evaluate on mini-validation dataset
- Evaluate on mini-testing dataset
- Evaluate on mini-causal-validation dataset
- Evaluate on mini-causal-testing dataset

Run as:
```bash
uv run -m scenetokens.train model=[model_name] paths=waymo
```

### Experiment 2: Waymo Causal Labeled
- Train on mini-causal dataset, which has labels for causal agents
- Evaluate on causal agents benchmark, which includes the following subsets:
    - *Original*: does not perturb the scene.
    - *Remove non-causal*: perturbs the scene by removing agents not labeled as causal.
    - *Remove causal*: perturbs the scene by removing agents labeled as causal.
    - *Remove non-causal-equal*: perturbs the scene by removing N non-causal agents, where N is the number of causal agents. This one is intended to be a softer benchmark for non-causality.
    - *Remove static*: perturbs the scene by removing agents whose motion is below a certain threshold.

Run as:
```bash
uv run -m scenetokens.train model=[model_name] paths=waymo_causal_labeled
```

### Experiment 3: Waymo Causal Unlabeled
- Train on mini dataset, which *does not* have causal agent labels.
- Evaluation follows that of *Experiment 2*.

Run as:
```bash
uv run -m scenetokens.train model=[model_name] paths=waymo_causal_unlabeled
```

# Model types
Current model versions are:
- *Wayformer*: which follows the architecture from UniTraj. To use it, run `train.py` with `model=wayformer`.
- *ScenetokensStudent*: which builds from Wayformer and adds a scenario tokenization layer. To use it, run `train.py` with `model=scenetokens_student`.
- *ScenetokensTeacher*: which builds from Wayformer and adds a scenario tokenization layer with causal awareness. To use it, run `train.py` with `model=scenetokens_teacher`.
