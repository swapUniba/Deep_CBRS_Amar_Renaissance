# Deep_CBRS_Amar_Renaissance

A fork from [Deep_CBRS_Amar_Revisited](https://github.com/cenzy/Deep_CBRS_Amar_Revisited) which follows the work from [Deep_CBRS_Amar](https://github.com/swapUniba/Deep_CBRS_Amar).

In this project, we experimented and evaluated different Graph Neural Network models to assess their performances in a recommendation task:

 - How well GNNs perform when used as collaborative features extractors in contrast to using precomputed embeddings obtained by relying on Knowledge Graph Embeddings (KGEs) literature? 
 - How can we effectively integrate GNNs in a hybrid recommender system which leverages both collaborative and content-based features, i.e. also including both textual content and properties of items?

For further details, refer to the [documentation](doc.pdf)

## Install

    pip install -r requirements.txt
    
## Dataset
In this work, we used MovieLens-1M, however, if for every dataset you'll need:

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
- **--exp_name**:  experiment name used in MLFlow that will encapsulate the runs .

For example, the following command runs a grid search on the _basic_ recommender system architecture with Graph Neural Networks:

    python src/experiment.py --exp_name myexp -e econfigs/basic-gnn.yaml
