## Install

      pip install -r requirements.txt

## Usage
Run an experiment with `experiment.py`
By default config.py and experiment.py will be used

    python src/experiment.py

Parameters:
- **-c** / **--config**: config input file, manages input parameters of the experiments;
- **-e** / **--experiments**: grid search experiment file, performs a grid search by overriding specified parameters in it with the ones in the config file;
- **--exp_name**:  experiment name used in MLFlow that will encapsulate the runs 


