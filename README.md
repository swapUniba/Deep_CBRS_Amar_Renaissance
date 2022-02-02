## Install

    pip install -r requirements.txt

## Usage
Run an experiment with `experiment.py`.
By default, config.yaml and experiments.yaml will be used for configuration.

    python src/experiment.py

The following parameters can be specified as well:
- **-c** / **--config**: config input file, manages input parameters of the experiments;
- **-e** / **--experiments**: grid search experiment file, performs a grid search by overriding specified parameters in it with the ones in the config file;
- **--exp_name**:  experiment name used in MLFlow that will encapsulate the runs .

For example, the following command runs a grid search on the _basic_ recommender system architecture with Graph Neural Networks:

    python src/experiment.py --exp_name myexp -e econfigs/basic-gnn.yaml
