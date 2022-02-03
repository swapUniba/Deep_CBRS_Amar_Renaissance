

# Deep CBRS Amar Renaissance

A fork from [Deep_CBRS_Amar_Revisited](https://github.com/cenzy/Deep_CBRS_Amar_Revisited) which follows the work from [Deep_CBRS_Amar](https://github.com/swapUniba/Deep_CBRS_Amar).

In this project, we experimented and evaluated different Graph Neural Network models to assess their performances in a recommendation task:

 - How well GNNs perform when used as collaborative features extractors in contrast to using precomputed embeddings obtained by relying on Knowledge Graph Embeddings (KGEs) literature? 
 - How can we effectively integrate GNNs in a hybrid recommender system which leverages both collaborative and content-based features, i.e. also including both textual content and properties of items?

For further details, refer to the [documentation](doc.pdf)

## Install
This repo requires at least Python 3.8

    pip install -r requirements.txt
    
## Dataset
### Get the dataset
In this work, we used MovieLens-1M, preprocessed files can be obtained with DVC:

    dvc pull dataset
### Use your own dataset
However, for every dataset you'll need:

 - CSV / TSV of ratings for train and test in the form `(user, item, liked / not liked)`

For Hybrid architectures:
 - JSON embeddings for content based features, respectively for user and item in the form:
	 - `[{"ID_OpenKE": <id_open_ke>, "profile_embedding": <embedding_in_list_form>}]`
	 - `[{"ID_OpenKE": <id_open_ke>, "embedding": <embedding_in_list_form>}]`

For knowledge-aware architectures:

 - Triples CSV / TSV in the form `(item, entity, relation)`

For knowledge-aware architectures:

 - Triples CSV / TSV in the form `(item, entity, relation)`

## Usage
Run an experiment with `experiment.py`.
By default, config.yaml and experiments.yaml will be used for configuration.

    python src/experiment.py

The following parameters can be specified as well:
- **-c** / **--config**: config input file, manages input parameters of the experiments;
- **-e** / **--experiments**: grid search experiment file, performs a grid search by overriding specified parameters in it with the ones in the config file;
- **--exp_name**:  experiment name used in MLFlow that will encapsulate the runs.

For example, the following command runs a grid search on the _basic_ recommender system architecture with Graph Neural Networks:

    python src/experiment.py --exp_name myexp -e econfigs/basic-gnn.yaml
**NOTE**: you need to pull the dataset to run our experiments [(↑)](#get-the-dataset) and the BERT embeddings for the Hybrid experiments:

    dvc pull embeddings

Refer to the [econfigs readme](./econfigs/README.md) for an explanation for every grid configuration of runs available. 
## Explore our experiments

If you want to see our results, download them with DVC (around 40 GiB)

    dvc pull mlruns
    
 User MLFlow to compare experiments with its UI:

     mlflow ui
     
Each run has its artifact folder where you can find the **trained weights**, the top 5/10 **calculated predictions** for the test set, **logs** and **config.yaml** to reproduce that exact run.
