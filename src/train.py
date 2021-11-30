from utilities.utils import read_bert_embeddings, read_ratings, matching_bert_emb_id
from models.model1 import run_model
from ruamel.yaml import YAML
from easydict import EasyDict

import logging
import sys

PARAMS_PATH = 'config.yaml'


def train(config):

    print(config.user_source)
    print(config.item_source)
    print(config.dest)
    print(config.prediction_dest)

    print('Reading BERT embeddings')
    user_embeddings, item_embeddings = read_bert_embeddings(config.user_source, config.item_source)
    print('Reading ratings')
    user, item, rating = read_ratings(config.ratings)
    print('Matching')
    X, y, dim_embeddings = matching_bert_emb_id(user, item, rating, user_embeddings, item_embeddings)

    print('Running model')
    model = run_model(X, y, dim_embeddings, epochs=25, batch_size=512)

    # creates a HDF5 file 'model.h5'
    model.save(config.dest + 'model.h5')


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    with open(PARAMS_PATH, 'r') as params_file:
        yaml = YAML()
        config = EasyDict(**yaml.load(params_file))

    train(config)