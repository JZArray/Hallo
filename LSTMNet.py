#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:45:22 2018

@author: felix
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.core import Dropout
from keras import optimizers
from keras.utils.multi_gpu_utils import multi_gpu_model

#from keras.callbacks import EarlyStopping


import keras
import matplotlib
import pandas as pd
import os

matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
from numpy import newaxis

class LSTMNet:
    def __init__(self):
        # Initalisierung des Netzes
        self.model = Sequential()
        self.backend = keras.backend.backend()
        self.history = None
        self.batch_size = None
        self.early_stopping = None

    def set_param(self, time_window, feature_length, batch_size, output, neurons, gpu_training, layer=1,
                  early_stopping=False, activation="sigmoid", dropout=0, kernel_initializer="random_uniform",
                  bias_initializer="zeros", stateful=False, optimiser="sdg"):
        '''In der set_param Methode wird das Neuronale Netz erstellt und die In-und Output-Dimensionen
        geklärt.
        input_shape_I:  Zeitdimension der Input Matrix, vorerst 1
        input_shape_II: Länge der Input-Features, hier die Länge der Spalten
        output:         Länge des Output-Vektors, hier die zu Vorhersagenden Zeitschritte'''
        # Netzwerk design
        unroll = False
        if time_window > 1:
            unroll = True

        for l in range(0, layer):
            if l == layer - 1:
                self.model.add(LSTM(units=neurons, input_shape=(time_window, neurons),
                                    batch_input_shape=(batch_size, time_window, feature_length),
                                    return_sequences=False, unroll=unroll, activation=activation,
                                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                    stateful=stateful))
                if dropout > 0:
                    self.model.add(Dropout(dropout))
            else:
                self.model.add(LSTM(units=neurons, input_shape=(time_window, neurons),
                                    batch_input_shape=(batch_size, time_window, feature_length),
                                    return_sequences=True, unroll=unroll, activation="tanh",
                                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                    stateful=stateful))
                if dropout > 0:
                    self.model.add(Dropout(dropout))

        self.model.add(Dense(output, activation="linear", use_bias=True, bias_initializer=bias_initializer))

        if optimiser == "sdg":
            opti = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        elif optimiser == "adam":
            opti = 'adam'
        elif optimiser == "rmsprop":
            opti = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

        if gpu_training:
            print("Multi GPU training")
            gpu_list = ["gpu(%d)" % i for i in range(2)]
            self.model = multi_gpu_model(model=self.model, gpus=2)
            self.model.compile(loss='mse', optimizer=opti, metrics=['mse', 'mae', 'mape'], context=gpu_list)
        else:
            self.model.compile(loss='mse', optimizer=opti, metrics=['mse', 'mae', 'mape'])

        self.batch_size = batch_size
        #if early_stopping:
        #    self.early_stopping = EarlyStopping(monitor='val_loss', patience=4)


    def train(self, train_x, train_y, epochs=10, batch=128):
        # trainieren des Netzwerkes mit einzelnen Batches
        self.history = self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch,
                                      verbose=0, shuffle=False)

    def train_multi(self, generator, validation, validation_steps, epochs, steps_per_epoch,
                    worker=1, use_multiprocessing=False):
        #trainieren und evaluieren des Netzwerkes mit generator
        self.history = self.model.fit_generator(generator, epochs=epochs, verbose=2,
                                                validation_data=validation,
                                                steps_per_epoch=steps_per_epoch,
                                                validation_steps=validation_steps,
                                                shuffle=False, use_multiprocessing=use_multiprocessing,
                                                workers=worker,max_queue_size=20)

    def evaluate_multi(self, generator, steps):
        #trainieren des netzwerkes mit multiplen Kernen
        self.history = self.model.evaluate_generator(generator, workers=10, steps=steps,
                                                     use_multiprocessing=False)

    def predict(self, x_set):
        # vorhersage machen
        prediction = self.model.predict(x_set)
        #prediction = np.reshape(prediction, (prediction.size,))
        return prediction

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def evaluate(self, test_x, test_y, verbosemode=0):
        # Evaluieren des Modells anhand eines test sets
        self.model.evaluate(test_x, test_y, verbose=verbosemode)

    def save_history(self, csv_path, name):
        history_path = csv_path + 'LSTM_Nets/' + keras.backend.backend() + '/' + name + '/'
        #history_path = 'LSTM_Nets/' + keras.backend.backend() + '/' + name + '/'

        os.makedirs(history_path, exist_ok=True)

        self.model.save(history_path + name + ".h5")

        # summarize history for mse
        plt.plot(self.getHistory().history['mean_absolute_error'], label="train")
        plt.plot(self.getHistory().history['val_mean_absolute_error'], label="validation")
        plt.title('model mean absolute error: with batch size ' + str(self.batch_size))
        plt.ylabel('mean absolute error')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.savefig(history_path + name + "_mae.png")
        plt.close("all")

        # summarize history for mape
        plt.plot(self.getHistory().history['mean_absolute_percentage_error'], label="train")
        plt.plot(self.getHistory().history['val_mean_absolute_percentage_error'], label="validation")
        plt.title('model val mean absolute percentage error: with batch size ' + str(self.batch_size))
        plt.ylabel('mean absolute percentage error')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.savefig(history_path + name + "_mape.png")
        plt.close("all")

        # summarize history for loss
        plt.plot(self.getHistory().history['loss'], label="train")
        plt.plot(self.getHistory().history['val_loss'], label="validation")
        plt.title('model loss: with batch size ' + str(self.batch_size))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.savefig(history_path + name + "_loss.png")
        plt.close("all")

        historyDataArray = [self.getHistory().history['loss'],
                            self.getHistory().history['val_loss'],
                            self.getHistory().history['mean_squared_error'],
                            self.getHistory().history['val_mean_squared_error'],
                            self.getHistory().history['mean_absolute_error'],
                            self.getHistory().history['val_mean_absolute_error'],
                            self.getHistory().history['mean_absolute_percentage_error'],
                            self.getHistory().history['val_mean_absolute_percentage_error']]

        historyData = pd.DataFrame(historyDataArray)
        historyData = historyData.transpose()

        historyData.columns = ['loss', 'val_loss',
                               'mse', 'val_mse',
                               'mae', 'val_mae',
                               'mape', 'val_mape']

        historyData.to_csv(history_path + name + "_history.csv", header=True)


        return historyData

    def getHistory(self):
        return self.history

    def getModel(self):
        return self.model

    def getBackend(self):
        return self.backend

