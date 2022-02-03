## Pre-computed BERT and Graph Embeddings Directory

Note that all the pre-computed embeddings can be downloaded using DVC [^1].
Please refer to the README.md in the root directory for details.

### Movielens-1M BERT Embeddings Directory Structure

- ```item-lastlayer.json```: items BERT embeddings obtained by summing up the word embeddings of the last layer of BERT;
- ```user-lastlayer.json```: users BERT embeddings obtained by the liked items BERT embeddings;
- ```item-lastlayer_nostopw.json```: items BERT embeddings, but without stop-words;
- ```user-lastlayer_nostopw.json```: users BERT embeddings.

Note that BERT word embedidngs are obtained using the plot of the movies.

### Movielens-1M Graph Embeddings Directories Structure

- ```user-item/```: directory containing graph embeddings for both users and items;
- ```user-item-properties/```: directory containing graph embeddings for users, items and properties.

Each graph embedding file is named as "{_D_}{_M_}.json", where _D_ denots the size of the embeddings (e.g. 768), and
_M_ denotes the Knowledge Graph Embeedings (KGE) model used (e.g. TransD).
