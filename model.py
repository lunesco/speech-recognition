import re

from joblib import dump, load
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

FILENAME = '../models/saved_model_'
N_ESTIMATORS = 500
RANDOM_STATE = 0
PATTERN = '([A-Z]+)_.*'


class Model:

    def __init__(self, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, splitted=False):
        self._model = RandomForestClassifier(n_estimators=n_estimators,
                                             random_state=random_state)
        self._splitted = splitted

    def get_model(self):
        return self._model

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X):
        if self._splitted:
            predicts = self._model.predict(X)
            core_predicts = []
            for predict in predicts:
                core_predict = re.findall(PATTERN, predict)[0]
                core_predicts.append(core_predict)
            prediction = stats.mode(core_predicts)[0][0]
        else:
            prediction = stats.mode(self.get_model().predict(X))[0][0]

        return prediction

    def score(self, X, y):
        good, alls = 0, 0
        for i, feat in enumerate(X):
            alls += 1
            if self.predict(X=feat) == y[i]:
                good += 1
        try:
            print(f"accuracy = {good / alls * 100}%")
            return good / alls
        except ZeroDivisionError:
            print("Cannot divide by zero! Bad value for 'alls'")

    def save(self, mode: str):
        dump(value=self._model, filename=FILENAME + mode + '_' + str(self._splitted))

    def load(self, mode: str):
        self._model = load(FILENAME + mode + '_' + str(self._splitted))
