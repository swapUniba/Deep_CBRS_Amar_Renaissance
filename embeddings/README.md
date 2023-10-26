## Pre-computed BERT and Graph Embeddings Directory

Note that all the pre-computed embeddings can be downloaded using DVC [^1].

### Movielens-1M BERT Embeddings Directory Structure

- ```item-lastlayer.json```: items BERT embeddings obtained by summing up the word embeddings of the last layer of BERT;
- ```user-lastlayer.json```: users BERT embeddings obtained by the liked items BERT embeddings;
- ```item-lastlayer_ns.json```: items BERT embeddings, but without stop-words;
- ```user-lastlayer_ns.json```: users BERT embeddings, but without stop-words.

Note that BERT word embedidngs are obtained using the plot of the movies.
Movielens-1M BERT embeddings can be downloaded as:

    dvc pull embeddings/bert

### Movielens-1M Graph Embeddings Directories Structure

- ```user-item/```: directory containing graph embeddings for both users and items;
- ```user-item-properties/```: directory containing graph embeddings for users, items and properties.

Each graph embedding file is named as "{_D_}{_M_}.json", where _D_ denotes the size of the embeddings (e.g. 768), and
_M_ denotes the Knowledge Graph Embeedings (KGE) model used (e.g. TransD).
Note that graph embeddings are obtained by relying on the user-item (or user-item-properties) graph.
Movielens-1M graph embeddings can be downloaded as:

    dvc pull embeddings/user-item
    dvc pull embeddings/user-item-properties
