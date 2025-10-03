# Scene-Tokens

## Installation

### Dependencies:
* [ScenarioCharacterization](https://github.com/navarrs/ScenarioCharacterization/)

Clone the Repository
```bash
mkdir tools
cd tools
git clone git@github.com:navarrs/ScenarioCharacterization.git
```

Verify Installation
```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run python -c "import characterization"
```

### Optional Dependencies:
* [ScenarioNet](https://scenarionet.readthedocs.io/en/latest/install.html)
* [MetaDrive](https://scenarionet.readthedocs.io/en/latest/install.html#install-metadrive)

Clone the Optional Repositories
```bash
mkdir tools
cd tools
git clone git@github.com:metadriverse/scenarionet.git
git clone git@github.com:metadriverse/metadrive.git
```

Verify Installation
```bash
uv run python -m metadrive.examples.profile_metadrive
uv run python -c "from scenarionet.common_utils import read_scenario; print(read_scenario)"
```

## Documentation

All documentation needed to use this repository is below:

* [Dataset Preparation](./docs/DATA_PREPARATION.md): Instructions to download the [Waymo Open Motion Dataset](https://waymo.com/open/) and prepare it for training using [ScenarioNet](https://scenarionet.readthedocs.io/en/latest/install.html) or a **custom** pre-processor.
* [Model Training](./docs/MODEL_TRAINING.md): Instructions to train models.
* [Experimental Design](./docs/EXPERIMENTS.md): Describes the current experimental setup, and how to launch training experiments for each setup.
* [Running Analysis](./docs/ANALYSIS.md)
* [Scenario Characterization](https://github.com/navarrs/ScenarioCharacterization/blob/main/docs/CHARACTERIZATION.md): Standalone scenario characterization instructions. NOTE: requires cloning or installing the [ScenarioCharacterization](https://github.com/navarrs/ScenarioCharacterization) package.
