from time import time

import numpy as np

from feature_extractor import Extractor
from model import Model
from reader import ALL_STUDENTS_TRAIN_SIZE, ONE_STUDENT_TRAIN_SIZE
from reader import DbReader

ALL_FOLDER_PATH = '../Pliki'
ONE_FOLDER_PATH = '../Pliki/266710'

TOP = 0.3
MID = 0.4
BOT = 0.3  # not used but initialized for completeness


def main(mode, splitted):
    # READ THE DATA
    if mode == 'one':
        reader = DbReader(ONE_FOLDER_PATH, mode=mode, train_size=ONE_STUDENT_TRAIN_SIZE)
    elif mode == 'all':
        reader = DbReader(ALL_FOLDER_PATH, mode=mode, train_size=ALL_STUDENTS_TRAIN_SIZE)
    else:
        raise Exception(f"Bad argument: mode = {mode}")

    train = reader.get_train()

    extractor = Extractor()
    train_feats_list, train_labels = [], []

    # FORMAT THE FILES
    for train_elem in train:
        label, array, rate = train_elem
        features = extractor.extract_features(signal=array, rate=rate)
        features_size = features.shape[0]

        train_feats_list.append(features)

        if splitted:
            top_count = round(TOP * features_size)
            mid_count = round(MID * features_size)
            bot_count = features_size - top_count - mid_count

            train_labels += top_count * [label + '_TOP']
            train_labels += mid_count * [label + '_MID']
            train_labels += bot_count * [label + '_BOT']
        else:
            train_labels += features_size * [label]

    train_feats = train_feats_list[0]
    for array in train_feats_list[1:]:
        train_feats = np.append(train_feats, array, axis=0)

    # TRAIN THE MODEL
    model = Model(n_estimators=500, splitted=splitted)

    t_start = time()
    model.fit(train_feats, train_labels)
    t_end = time()

    print(f"Fitting took {t_end - t_start}s.")

    # SAVE THE MODEL
    model.save(mode=mode)


if __name__ == "__main__":
    main(mode='one', splitted=True)
