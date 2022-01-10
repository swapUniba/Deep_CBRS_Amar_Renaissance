import os
import subprocess
import logging
import numpy as np
import pandas as pd

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def top_k_predictions(predictions, users, items, k=5):
    """
    Compute the Top-K suggested items for each user, given prediction scores.

    :param predictions: The prediction scores of users and items.
    :param users: The original users identifiers.
    :param items: The original items identifiers.
    :param k: The K parameter.
    :return: The top K predictions Pandas data frame, with the original users and items identifiers.
    """
    # Build the predictions data frame
    # Note mapping user and item ids back to their identifiers
    df = pd.DataFrame()
    df['users'] = users[predictions[:, 0].astype(np.int64)]
    df['items'] = items[predictions[:, 1].astype(np.int64) - len(users)]
    df['scores'] = predictions[:, 2]
    predictions = df.sort_values(by=['users', 'scores'], ascending=[True, False])

    # Compute the top K predictions based on scores
    top_k_scores = pd.DataFrame()
    for u in set(predictions['users']):
        p = predictions.loc[predictions['users'] == u]
        top_k_scores = top_k_scores.append(p.head(k))
    return top_k_scores


def top_k_metrics(test_filepath, predictions_path):
    try:
        if not os.path.isdir(predictions_path):
            raise RuntimeError("Invalid predictions path specified. Unable to run evaluator.")

        for root, dirs, files in os.walk(predictions_path):
            filtered_files = list(filter(lambda x: x.startswith("predictions"), files))
            print(filtered_files)
            num_filtered_files = len(filtered_files)
            print(num_filtered_files)
            if num_filtered_files == 1:
                cutoff = str(root)[root.rfind(os.sep):].split("_")[1]
                print(cutoff)
                results_filename = os.sep.join([root, "results.tsv"])
                mimir_output = subprocess.call(["java", "-jar", "binaries/mimir.jar",
                                                "-holdout",
                                                "-cutoff", cutoff,
                                                "-test", test_filepath,
                                                "-predictions", os.sep.join([root, files[0]]),
                                                "-results", results_filename])
                logger.info(mimir_output)
            elif num_filtered_files > 1:
                cutoff = str(root)[root.rfind(os.sep):].split("_")[1]
                results_filename = os.sep.join([root, "results.tsv"])
                mimir_output = subprocess.call(["java", "-jar", "binaries/mimir.jar",
                                                "-cv",
                                                "-folds",
                                                str(num_filtered_files),
                                                "-cutoff", cutoff,
                                                "-test", test_filepath,
                                                "-predictions", os.sep.join([root, "predictions_%s.tsv"]),
                                                "-results", results_filename])
                logger.info(mimir_output)
    except RuntimeError as e:
        logger.exception(e)
