## Datasets Directory

Note that all the data and preprocessed datasets can be downloaded using DVC [^1].

### Movielens-1M

The _Movielens-1M_ dataset contains approximatively 1 milion positive and negative ratings of users to films.
Additionally, relationships between movies and some of their properties are also included.
Note that, there are two relation types settings for the relationships between movies and properties:
1. subject, director, starring, writer, language, editing, narrator;
2.  subject, director, starring, writer,  language, editing, cinematography, musicComposer, country, producer, basedOn.

The dataset can be downloaded as:

    dvc pull datasets/movielens

#### Directory Structure

- ```train2id.tsv```: the training data composed of binary ratings between users and movies IDs
(0:negative and 1:positive).
- ```test2id.tsv```: the testing data composed of binary ratings between users and movies IDs.
Note that every user and item ID appear also in the training data. Of course this split does contain additional ratings.
- ```props2id-1relconf.tsv```: the relationships between movies and properties IDs, with the relation type setting (1).
- ```props2id-2relconf.tsv```: the relationships between moveis and properties IDs, with the relation type setting (2).
- ```legacy/```: a directory containing legacy information about the dataset, i.e. mappings between item and properties
IDs, and URIs of objects in DBpedia; mappings of users and movies IDs w.r.t. the original Movielens-1M dataset; and
other relation type settings. Please refer to https://github.com/cenzy/Deep_CBRS_Amar_Revisited for additional
information about the content of this directory.

[^1]: Data Version Control - https://github.com/iterative/dvc
