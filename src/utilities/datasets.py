import csv
import numpy as np
import pandas as pd


def load_ratings(filepath):
    users, items, ratings = [], [], []
    with open(filepath) as fp:
        csv_reader = csv.reader(fp, delimiter='\t')
        for row in csv_reader:
            users.append(int(row[0]))
            items.append(int(row[1]))
            ratings.append(int(row[2]))
    return users, items, ratings
