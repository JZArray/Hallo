#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 09:17:09 2018

@author: felix
"""

import pandas as pd
import os
import datetime as dt
import time
import argparse
#import numpy as np
import logging

parser = argparse.ArgumentParser(description="Trainiere ein neues LSTM Netz",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-E', '--num-epochs', type=int, default=150,
                    help='Anzahl der Trainingsepochen')

parser.add_argument('-L', '--num-layer', type=int, default=1,
                    help='Anzahl der Netzeben')

parser.add_argument('-N', '--num-neurons', type=int, default=200,
                    help='Anzahl der Neuronen je LSTM-Layer')

parser.add_argument('-B', '--batch-size', type=int, default=2**10,
                    help='Batch Größe')

parser.add_argument('-FG', '--fit-generator', default=True,
                    help='Training per Generator oder eigener Fitfunktion')

parser.add_argument('-F', '--front-node', type=int, default=3, choices=[1,2,3],
                    help='Wo und mit welchen Daten soll die Berechnung ausgeführt werden.')
                    
parser.add_argument('-P', '--path',
                    help='Alternativ zum Front-Node Argument kann auch der Path manulell '
                         'übergeben werden.')

parser.add_argument('-TP', '--predicting-timesteps', type=int, default=5,
                    help='Anzahl der Zeitschritte die vorrausgesagt werden sollen')

parser.add_argument('-TW', '--window-timesteps', type=int, default=1,
                    help='Anzahl der Zeitschritte die bei der Vorhersage mit berücksichtigt werden sollen')

parser.add_argument('-PB', '--progress-tune', type=int, default=15,
                    help='Schrittgröße der Progressbar')

parser.add_argument('-FL', '--file-length', type=int, default=None,
                    help='Wieviele Dateien bei der Erstellung des Netzes berücksichtig werden')

parser.add_argument('-G', '--GPU-training', default=True,
                    help='GPU Training')

parser.add_argument('-GR', '--grid', type=int, choices=[1, 2, 3, 4, 5, 6, 7],
                    help='Vergleiche Hyperparametervariation. \n'
                         '1:  Variation der Neuronen und Epochenanzahl 3-20 und 50/100/150 \n'
                         '2:  Variation der Neuronen und des Schichten \n'
                         '3:  Variation der Batchgröße \n'
                         '4:  Variation der Variation der Epochenlänge 50-4000 \n'
                         '5:  Variation der Variation des Neuronen Dropout von 0- 0.5 \n'
                         '6:  Untersuchung der Sinusdatasets \n'
                         '7:  keine Variation\n')

parser.add_argument('-PT', '--prognosis-type', default="I",
                    help='Welche PQ-Größe soll Prognostiziert werden? \n'
                         'I - Strom (rms_i0, rms_i1, rms_i2, rms_i3)\n'
                         'U - Spannung (rms_u0, rms_u1, rms_u2, rms_u3)\n'
                         'F - Frequenz (freq0)\n'
                         'Es kann auch ein eigene Größe übergeben zB: rms_i1')

parser.add_argument('-WT', '--window-type', default="U",
                    help='Welche PQ-Größe soll der Prognose zu Grunde gelegt werden? \n'
                         'I - Strom (rms_i0, rms_i1, rms_i2, rms_i3)\n'
                         'U - Spannung (rms_u0, rms_u1, rms_u2, rms_u3)\n'
                         'F - Frequenz (freq0)')

parser.add_argument('-GPU', '--GPU', type=int, default=1, choices=[1,2],
                    help='Welche GPU')

parser.add_argument('-D', '--dropout', type=float, default=0.0,
                    help='Dropout')

parser.add_argument('-SF', '--stateful', default=False,
                    help='Stateful')

parser.add_argument('-O', '--optimiser', default='adam', choices=['adam', 'sgd', 'RMSprop'],
                    help='Wahl des Optimierers')

parser.add_argument('-AN', '--name', default='',
                    help='Nameszusatz')

from load_data import LoadData as LoadData
import LSTMNet as net
import DataGenerator as generator

def main(data_path, csv_path, p_timesteps, w_timesteps=1, layer=1, epochs=10,
         batch_size=128, neurons=200, gpu_training=False, prognosis_type="I", norm="ARENA", file_length=None,
         activation="sigmoid", dropout=0, kernel_initializer="random_uniform", bias_initializer="zeros",
         stateful=False, epochs_step_training=0, optimiser='sdg', add_name='', window_type='I'):
    """"Main Funktion des neuronalen Netzes."""

    start_time = dt.datetime.now()
    print("Start des Programms um: " + str(dt.datetime.now()))

    files = os.listdir(data_path)
    try:
        files.remove('.DS_Store')
    except ValueError:
        pass

    # Um das Trainig zu verkürzen kann dem Skrift ein File_Length Faktoritgegeben werden
    if (file_length!=None):
        files = files[:file_length]

    boolean_norm_data = True
    file_type = "hdf"
    data_file_length = 86400

    # Configuration
    # Zu beachtende Spalten:
    # Die ARENA Daten haben nur Strom un Spannung am Trafo
    columns = ['rms_i0', 'rms_i1', 'rms_i2', 'rms_i3']

    if prognosis_type == "I":
        prediction_columns = ['rms_i0', 'rms_i1', 'rms_i2', 'rms_i3']

    elif prognosis_type == "F":
        prediction_columns = ['freq0']

    elif prognosis_type == "U":
        prediction_columns = ['rms_u1', 'rms_u2', 'rms_u3']

    elif prognosis_type == "sinus":
        prediction_columns = ['A']
        columns = ['A']  # Sinus Testdaten
        boolean_norm_data = True
        file_type = "csv"
        data_file_length = 200

    else:
        prediction_columns = [prognosis_type]


    if window_type == "I":
        columns = ['rms_i0', 'rms_i1', 'rms_i2', 'rms_i3']

    elif window_type == "F":
        columns = ['freq0']

    elif window_type == "U":
        columns = ['rms_u1', 'rms_u2', 'rms_u3']

    else:
        columns = [window_type]

    print(data_file_length)
    # Erstellung das KNN
    model = net.LSTMNet()
    model.set_param(time_window=w_timesteps, feature_length=len(columns), batch_size=batch_size,
                    output=p_timesteps * len(prediction_columns), layer=layer, neurons=neurons,
                    gpu_training=gpu_training, activation=activation, dropout=dropout,
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, stateful=stateful,
                    optimiser=optimiser)

    # unterteile in Trainigsdaten, Evaluierungs und Testdaten
    trainLength = int(len(files)*0.6) # werden zum Training genutzt
    evaluLength = int(len(files) * 0.2) # werden zum evaluieren des Netzes Keras intern genutzt
    testLength = int(len(files) * 0.2) # werden später zum testen genutzt

    # PrintOut um die Struktur des erstellten Modells zu zeigen
    model.getModel().summary()
    print("Hyperparameter: \n - LSTM Layer: \t\t\t" + str(layer) + "\n - Trainigs Epochen: \t\t" + str(epochs) +
          "\n - Batch Size: \t\t\t" + str(batch_size) + "\n - Prognosezeitschritte: \t" + str(p_timesteps) +
          "\n - Zeitfenster: \t\t" + str(w_timesteps), "\n - Stateful: \t\t\t" + str(stateful),
          "\n - Activation: \t\t\t" + str(activation), "\n - dropout: \t\t\t" + str(dropout),
          "\n - Kernel Initializer: \t\t" + str(kernel_initializer), "\n - Datengröße: \t\t\t" + str(file_length),
          "\n - Prognoseart: \t\t" + prognosis_type, "\n - Optimierer: \t\t\t" + optimiser)
    print("_" * 65)

    name =  dt.datetime.strftime(dt.datetime.now(), '%Y%m%d')+ "_"+ add_name + "_" + prognosis_type + "_"

    # Abfrage ob mit dem Datengenerator gearbeitet werden soll oder nicht
    model.train_multi(generator.generate_data_all_files(files=files[:trainLength], columns=columns,
                                                prediction_columns=prediction_columns, prediction_timeteps=p_timesteps,
                                                window_timeteps=w_timesteps, batch_size=batch_size, type="training",
                                                progress_tune=args.progress_tune, LoadData=LoadData, data_path=data_path,
                                                csv_path=csv_path, norm=norm, normData=boolean_norm_data,filetype=file_type),
                      generator.generate_data_all_files(files=files[trainLength:trainLength+evaluLength], columns=columns,
                                                prediction_columns=prediction_columns, prediction_timeteps=p_timesteps,
                                                window_timeteps=w_timesteps, batch_size=batch_size, type="validation",
                                                progress_tune=args.progress_tune, LoadData=LoadData, data_path=data_path,
                                                csv_path=csv_path, norm=norm, normData=boolean_norm_data,filetype=file_type),
                      epochs=epochs, steps_per_epoch=int(data_file_length/batch_size)*len(files[:trainLength]),
                      validation_steps=int(data_file_length/batch_size)*len(files[trainLength:trainLength+evaluLength]),
                      worker=1, use_multiprocessing=False)

    logging.info("Ende des Trainings um: " + str(dt.datetime.now()))
    print("Ende des Trainings um: " + str(dt.datetime.now()))
    name += "l" + str(layer) + "n" + str(neurons) + "e" + str(epochs) + "bs" + str(batch_size)

    ###############################################################################
    # Ab hier ist das Training beendet und die Dokumentation des Neuronalen Netzes wird übernommen
    history_data = model.save_history(csv_path, name)
    if prognosis_type == "sinus":
        overview_name = "NN_SinusError.csv"

    else:
        overview_name = "NN_Overview.csv"

    old_overview = pd.read_csv(csv_path + overview_name)
    try:
        old_overview = old_overview.drop('Unnamed: 0', axis=1)
    except KeyError:
        pass

    end_time = dt.datetime.now()
    runtime = str(end_time - start_time)

    ov_I = pd.DataFrame(data=[[name, runtime, start_time, end_time, optimiser, dropout, file_length, columns, prediction_columns,
                               epochs, neurons, layer, batch_size, w_timesteps, p_timesteps, files[-testLength:]]],
                        columns=['name', 'runtime', 'starttime', 'endtime', 'optimiser', 'dropout', 'file_size', 'feature',
                                 'prediction', 'epochs', 'neuronen', 'layer', 'batch_size', 'history_window', 'prediction_window',
                                 'test_daten'])

    ov_II = pd.DataFrame(data=[history_data[-1:].values[0]],
                         columns=['loss', 'val_loss', 'mse', 'val_mse', 'mae', 'val_mae',
                                  'mape', 'val_mape'])

    overview = pd.concat([ov_I, ov_II], axis=1)
    new_overview = pd.concat([old_overview, overview], axis=0, ignore_index=True)
    new_overview[['name', 'runtime', 'starttime', 'endtime', 'optimiser','dropout', 'file_size', 'feature',
                  'prediction', 'epochs', 'neuronen', 'layer', 'batch_size', 'history_window', 'prediction_window',
                  'test_daten', 'loss', 'val_loss', 'mse', 'val_mse', 'mae', 'val_mae', 'mape', 'val_mape',
                  'mse_test', 'mae_test', 'mape_test', 'rmse_test']].to_csv(csv_path + overview_name,
                                                                            float_format='%.4f')

    return model, name


if __name__ == '__main__':
    args = parser.parse_args()
    # Falls das Skript verzögert werden muss
    #time.sleep(4000)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    data_path = ""
    csv_path = ""

    if (args.front_node) == 1:
        '''Config: Server FrondNode mit ARENA Daten'''
        data_path = "/share/Shared/DATEN_AUS_ARENA/Trafo1/h5/"
        csv_path = "/home/fst/masterarbeit/Share/Code/src/Felix/Data/"

    elif int(args.front_node) == 2:
        '''Config: Sinus Testdaten '''
        data_path = '/share/Shared/SinusData/'
        #data_path = "/Users/felixstoeckmann/Documents/Studium/Master/Masterarbeit/TestDaten/TestData/"
        csv_path = "/share/Shared/masterarbeit/Share/Code/src/Felix/Data/"

    elif int(args.front_node) == 3:
        '''Config: Server CmputeNode '''
        print('Config: Server CmputeNode')
        data_path = "/share/Shared/DATEN_AUS_ARENA/Trafo1/h5/"
        csv_path = "/share/Shared/masterarbeit/Share/Code/src/Felix/Data/"
        print(data_path)

    if int(args.grid) == 1:
        for n in[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            # Neuronen Grid
            for i in [50, 100, 150]:
                # Epochen grid
                model, name = main(data_path, csv_path, p_timesteps=args.predicting_timesteps, w_timesteps=args.window_timesteps,
                               layer=args.num_layer, epochs=i, batch_size=args.batch_size,
                               neurons=n, prognosis_type=args.prognosis_type, file_length=args.file_length,
                               dropout=float(args.dropout), stateful=args.stateful,
                               epochs_step_training=args.epochs_step_training,
                               optimiser=args.optimiser, add_name=args.name, window_type=args.window_type)

    elif int(args.grid) == 2:
        for n in [10, 20, 30]:
            # Neuronen Grid
            for i in [1, 2, 3, 4]:
                # Layer grid
                model, name = main(data_path, csv_path, p_timesteps=args.predicting_timesteps,
                                   w_timesteps=args.window_timesteps,
                                   layer=i, epochs=args.num_epochs, batch_size=args.batch_size,
                                   neurons=n, prognosis_type=args.prognosis_type, file_length=args.file_length,
                                   dropout=float(args.dropout), stateful=args.stateful,
                                   epochs_step_training=args.epochs_step_training,
                                   optimiser=args.optimiser, add_name=args.name, window_type=args.window_type)

    elif int(args.grid) == 3:
        for n in [64, 128, 256, 512, 1024]:
            # Neuronen Grid
            for i in [1]:
                # Epochen grid
                model, name = main(data_path, csv_path, p_timesteps=args.predicting_timesteps,
                                   w_timesteps=args.window_timesteps,
                                   layer=i, epochs=args.num_epochs, batch_size=n,
                                   neurons=args.num_neurons, prognosis_type=args.prognosis_type, file_length=args.file_length,
                                   dropout=float(args.dropout), stateful=args.stateful,
                                   epochs_step_training=args.epochs_step_training,
                                   optimiser=args.optimiser, add_name=args.name, window_type=args.window_type)

    elif int(args.grid) == 4:
        for n in[20]:
            # Spezischisches Suchen nach dem Minumum in der Epochenanzahl
            for i in [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 4000]:
                # Epochen grid
                model, name = main(data_path, csv_path, p_timesteps=args.predicting_timesteps, w_timesteps=args.window_timesteps,
                               layer=args.num_layer, epochs=i, batch_size=args.batch_size,
                               neurons=n, prognosis_type=args.prognosis_type, file_length=args.file_length,
                               dropout=float(args.dropout), stateful=args.stateful,
                               epochs_step_training=args.epochs_step_training,
                               optimiser=args.optimiser, add_name=args.name, window_type=args.window_type)

    elif int(args.grid) == 5:
        # Gridsearch zur untersuchung des Neuronen Dropout
        for i in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
            # Dropout grid
            model, name = main(data_path, csv_path, p_timesteps=args.predicting_timesteps, w_timesteps=args.window_timesteps,
                         layer=args.num_layer, epochs=args.num_epochs, batch_size=args.batch_size,
                         neurons=args.num_neurons, prognosis_type=args.prognosis_type, file_length=args.file_length,
                         dropout=float(i), stateful=args.stateful, epochs_step_training=args.epochs_step_training,
                         optimiser=args.optimiser,add_name=args.name+str(i), window_type=args.window_type)

    elif int(args.grid) == 6:
        """Elif für eine variation des Sinus Noise Datasets"""
        for n in ['0.00','0.05','0.10','0.15','0.20','0.30','0.50','0.80']:
            new_data_path = data_path + "noise_" + n +"/"
            # Neuronen Grid
            for i in [50, 80, 100, 150]:
                print(n +" : " + str(i))
                # Epcohen grid
                model, name = main(new_data_path, csv_path, p_timesteps=args.predicting_timesteps, w_timesteps=args.window_timesteps,
                               layer=args.num_layer, epochs=i, batch_size=args.batch_size,
                               neurons=args.num_neurons, prognosis_type=args.prognosis_type, file_length=args.file_length,
                               dropout=float(args.dropout), stateful=args.stateful,
                               epochs_step_training=args.epochs_step_training,norm="sinus",
                               optimiser=args.optimiser, add_name=n, window_type=args.window_type)

    else:
        model, name = main(data_path, csv_path, p_timesteps=args.predicting_timesteps, w_timesteps=args.window_timesteps,
                         layer=args.num_layer, epochs=args.num_epochs, batch_size=args.batch_size,
                         neurons=args.num_neurons, prognosis_type=args.prognosis_type, file_length=args.file_length,
                         dropout=float(args.dropout), stateful=args.stateful, epochs_step_training=args.epochs_step_training,
                         optimiser=args.optimiser,add_name=args.name, window_type=args.window_type)

    #测试是否有反应
