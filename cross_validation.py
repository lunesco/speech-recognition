from json import dumps
from time import time
from warnings import simplefilter

import numpy as np

from feature_extractor import Extractor
from learning_script import ALL_FOLDER_PATH, ONE_FOLDER_PATH, MID, TOP
from model import Model
from reader import ALL_STUDENTS_TRAIN_SIZE, ONE_STUDENT_TRAIN_SIZE
from reader import DbReader

simplefilter(action='ignore')


def main(mode, splitted):
    t_start = time()

    # READ THE DATA
    if mode == 'one':
        reader = DbReader(ONE_FOLDER_PATH, mode=mode, train_size=ONE_STUDENT_TRAIN_SIZE)
    elif mode == 'all':
        reader = DbReader(ALL_FOLDER_PATH, mode=mode, train_size=ALL_STUDENTS_TRAIN_SIZE)
    else:
        raise Exception(f"Bad argument mode: {mode}")

    winlens = np.arange(0.02, 0.03, 0.001)
    winsteps = np.arange(0.01, 0.02, 0.001)
    nffts = list(range(1700, 2050, 50))
    ns = list(range(1, 15, 1))

    train = reader.get_train()
    valid = reader.get_valid()
    extractor = Extractor()

    # PREPERE THE FEATURES
    best_score = 0
    best_params = {}
    for winlen in winlens:
        for winstep in winsteps:
            for nfft in nffts:
                for n in ns:
                    train_feats_list, train_labels = [], []
                    valid_feats_list, valid_labels = [], []

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

                    for valid_elem in valid:
                        label, array, rate = valid_elem

                        features = extractor.extract_features(signal=array, rate=rate, winlen=winlen, winstep=winstep,
                                                              nfft=nfft, n=n)
                        valid_feats_list.append(features)
                        valid_labels += [label]

                    # TRAIN THE MODEL
                    model = Model(n_estimators=500, splitted=splitted)
                    model.fit(train_feats, train_labels)

                    # EVAL
                    print(f"\nwinlen = {winlen}, winstep = {winstep}, nfft = {nfft}, n = {n}")
                    score = model.score(valid_feats_list, valid_labels)

                    if score > best_score:
                        best_score = score
                        best_params['winlen'] = winlen
                        best_params['winstep'] = winstep
                        best_params['nfft'] = nfft
                        best_params['n'] = n

                print()
            print()
        print()
    t_end = time()

    print(f"Best score: {best_score * 100}%")
    print(f"Best params: {dumps(best_params)}")
    print(f"It took {t_end - t_start}s.")


if __name__ == "__main__":
    main(mode='all', splitted=False)
