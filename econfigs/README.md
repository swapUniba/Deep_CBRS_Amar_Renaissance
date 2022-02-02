# Experiments Configuration Files

In this directory there are several examples of experiments configuration files:
- ```basic-kge```: Basic architecture with user-item and user-item-properties graph embeddings, and with BERT
embeddings.
- ```hybrid-kge```: Hybrid architectures (both entity-based and feature-based) with user-item and
user-item-properties graph embeddings, and with BERT embeddings.
- ```basic-gnn```: Basic architecture with user-item embeddings given by Graph Neural Networks.
- ```hybrid-gnn```: Hybrid architecture (feature-based) with user-item embeddings given by Graph Neural Networks,
and with BERT embeddings.
- ```basic-gnn-uip-1relconf```: Basic architecture with user-item embeddings given by Graph Neural Networks, using the
user-item-properties graph with the first relation types setting.
- ```basic-gnn-uip-2relconf```: Basic architecture with user-item embeddings given by Graph Neural Networks, using the
user-item-properties graph with the second relation types setting.
- ```hybrid-gnn-uip-1relconf```: Hybrid architectures (both entity-based and feature-based) with user-item embeddings
given by Graph Neural Networks, using the user-item-properties graph with the first relation types setting; and with
BERT embeddings.
- ```hybrid-gnn-uip-2relconf```: Hybrid architectures (both entity-based and feature-based) with user-item embeddings
given by Graph Neural Networks, using the user-item-properties graph with the second relation types setting; and with
BERT embeddings.
- ```hybrid-gnn-tweaks```: Hybrid architecture (feature-based) with Attention and Residual tweaks, with user-item
embeddings given by Graph Neural Networks, and with BERT embeddings.
- ```hybrid-gnn-tweaks-uip-2relconf```: Hybrid architecture (feature-based) with Attention and Residual tweaks, with
user-item  embeddings given by Graph Neural Networks, using the user-item-properties graph with the second relation
types setting; and with BERT embeddings.

Note that the all the missing parameters are assumed to be equal to the default values in ```config.yaml``` at the root
directory level. 
