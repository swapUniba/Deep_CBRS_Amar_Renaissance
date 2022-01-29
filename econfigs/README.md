# Experiments Configuration Files

In this directory there are four examples of experiments configuration files:
- ```basic-kge.yaml```: Basic architecture with user-item and user-item-properties graph embeddings, and with BERT embeddings.
- ```hybrid-kge.yaml```: Hybrid architectures (both entity-based and feature-based) with user-item and user-item-properties graph embeddings, and with BERT embeddings.
- ```basic-gnn.yaml```: Basic architecture with user-item embeddings given by Graph Neural Networks.
- ```hybrid-gnn.yaml```: Hybrid architecture (feature-based) with user-item embeddings given by Graph Neural Networks, and with BERT embeddings.

Note that the all the missing parameters are assumed to be equal to the default values in ```config.yaml``` at the root directory level. 
