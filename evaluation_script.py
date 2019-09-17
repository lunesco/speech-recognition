import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from feature_extractor import Extractor
from learning_script import ALL_FOLDER_PATH, ONE_FOLDER_PATH
from model import Model
from reader import ALL_STUDENTS_TRAIN_SIZE, ONE_STUDENT_TRAIN_SIZE
from reader import DbReader


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          only_occuring_labels=False):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Only use the labels that appear in the data, which also sorts the labels
    if only_occuring_labels:
        if type(classes) is list:
            classes = unique_labels(y_true, y_pred)

        elif type(classes) is np.ndarray:
            classes = classes[unique_labels(y_true, y_pred)]

        else:
            raise Exception(f"Not supported type of classes: {type(classes)}. Use list or numpy.ndarray.")

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def main(mode, splitted):
    # READ THE DATA
    if mode == 'one':
        reader = DbReader(ONE_FOLDER_PATH, mode=mode, train_size=ONE_STUDENT_TRAIN_SIZE)
    elif mode == 'all':
        reader = DbReader(ALL_FOLDER_PATH, mode=mode, train_size=ALL_STUDENTS_TRAIN_SIZE)
    else:
        raise Exception(f"Bad argument mode: {mode}")

    valid = reader.get_valid()

    extractor = Extractor()
    valid_feats_list, valid_labels = [], []

    # LOAD THE MODEL
    model = Model(splitted=splitted)
    model.load(mode)

    # PREPERE THE FEATURES
    for valid_elem in valid:
        label, array, rate = valid_elem

        features = extractor.extract_features(signal=array, rate=rate)
        valid_feats_list.append(features)
        valid_labels += [label]

    model.score(valid_feats_list, valid_labels)

    predict = []
    classes = ['OTWORZ', 'ZAMKNIJ', 'GARAZ', 'ZROB', 'NASTROJ', 'WLACZ', 'WYLACZ', 'MUZYKE', 'SWIATLO', 'ZAPAL',
               'PODNIES', 'ROLETY', 'TELEWIZOR']

    for elem in valid_feats_list:
        predict.append(model.predict(X=elem))

    plot_confusion_matrix(y_true=valid_labels, y_pred=predict, classes=classes, normalize=True,
                          title='Normalized confusion matrix for commands', only_occuring_labels=True)
    plt.show()


if __name__ == "__main__":
    main(mode='one', splitted=True)
