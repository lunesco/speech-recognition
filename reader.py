import copy
import os
import re

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from sklearn.preprocessing import minmax_scale

PATTERN = '^([0-9]{6})_(2[1-6])_([KMkm])_([0-9.]*)_([0-9]).([a-z]{3})$'
ONE_STUDENT_TRAIN_SIZE = 0.5
ALL_STUDENTS_TRAIN_SIZE = 0.75


class DbReader:
    def __init__(self, path, mode, train_size):
        self.path = path
        self.train_size = train_size

        if mode == 'one':
            self.train, self.valid = self.read_one_student(self.path)
        elif mode == 'all':
            self.train, self.valid = [], []
            self.read_all_students()
        else:
            raise Exception(f"Bad argument: mode cannot be '{mode}'")

    def read_all_students(self):
        for directory in os.listdir(self.path):  # one subfolder in directory
            train, valid = self.read_one_student(os.path.join(self.path, directory))
            self.train += train
            self.valid += valid

    def read_one_student(self, path):
        wav_files, txt_files = {}, {}
        for filename in os.listdir(path):  # single file in folder
            filename = os.fsdecode(filename)
            if len(filename) <= 10:
                continue
            [(index, age, sex, time, number, extension)] = re.findall(PATTERN, filename)

            if extension == 'wav':
                wav_files[int(number)] = filename
            elif extension == 'txt':
                txt_files[int(number)] = filename
            else:
                raise Exception(f"Bad file extension: '{extension}'")

        return self._read_files(wav_files, txt_files, path)

    def _read_files(self, wav_files, txt_files, path):
        frames = {}
        signals = {}

        for key in txt_files.keys():
            with open(os.path.join(path, txt_files[key]), 'r') as fd:
                frames[key] = pd.read_csv(fd, sep='\t', header=None, float_precision='round_trip')

        rates = {}
        for key in wav_files.keys():
            rates[key], signals[key] = wav.read(os.path.join(path, wav_files[key]))  # rate, data

            if len(signals[key].shape) > 1:
                signals[key] = np.mean(signals[key], axis=1)

            signals[key] = minmax_scale(signals[key], feature_range=(-1, 1), axis=0).astype(np.float32)

        train_frames, train_signals, train_rates = copy.deepcopy(frames), copy.deepcopy(signals), copy.deepcopy(rates)
        valid_frames, valid_signals, valid_rates = copy.deepcopy(frames), copy.deepcopy(signals), copy.deepcopy(rates)

        i = 0
        length = len(frames)
        for key in frames.keys():
            if i < round(self.train_size * length):
                del valid_frames[key], valid_signals[key], valid_rates[key]
            else:
                del train_frames[key], train_signals[key], train_rates[key]
            i += 1

        train_data = self._split_signals(train_frames, train_signals, train_rates)
        valid_data = self._split_signals(valid_frames, valid_signals, valid_rates)

        return train_data, valid_data

    @staticmethod
    def _split_signals(frames, signals, rates):
        splitted_signal = []
        for key in frames.keys():
            for ind, row in frames[key].iterrows():
                t_start = float(row[0])
                t_end = float(row[1])
                command = row[2]

                splitted_signal.append([
                    command,
                    signals[key][round(t_start * rates[key]):round(t_end * rates[key])],
                    rates[key]
                ])

        return splitted_signal

    def get_train(self):
        return self.train

    def get_valid(self):
        return self.valid
