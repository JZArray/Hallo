#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 09 13:46:42 2018

@author: felix
"""

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import pandas as pd
import os
import keras

from matplotlib import pyplot as plt
from load_data import LoadData as LoadData
import DataGenerator as generator
import Metrics as mt
import PlotData as pt
import argparse

parser = argparse.ArgumentParser(description="Trainiere ein neues LSTM Netz",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-GPU', '--GPU', type=int, default=1,
                    help='Auf welcher GPU sollen die berechnungen ausgeführt werden')

parser.add_argument('-T', '--evalu-type', default="",
                    help='Evalurations Type: sinus oder multi')

parser.add_argument('-N', '--name', default="",
                    help='Evalurations Type')

parser.add_argument('-I', '--sum-evatuation', type=int, default=0,
                    help='Übergibt den Index der Overviewdatei ab der die Netze Evaluiert werden sollen.')

parser.add_argument('-NN', '--net-name',
                    help='Name des zu evaluierenden Netzes')

parser.add_argument('-P', '--plots', type=int, default=1,
                    help='End Integer der zu evaluierenden Netze, wenn 1 dann wird geplottet sonst nicht')


def load_model(file_path):
    model = keras.models.load_model(file_path)
    return model


def load_Overview(csv_path=None, NN_Overview_Name="NN_Overview.csv",NN_Overview=None):
    print("[INFO:] Load Settings")
    if NN_Overview==None:
        NN_Overview = pd.read_csv(csv_path + NN_Overview_Name)

        try:
            NN_Overview = NN_Overview.drop('Unnamed: 0', axis=1)
        except KeyError:
            pass

    NN_Overview = NN_Overview.set_index("name")

    return NN_Overview


def load_settings_form_Overview(NN_Overview, net_name):
    print("[INFO:] Load Data")
    files = os.listdir(data_path)
    try:
        files.remove('.DS_Store')
    except ValueError:
        pass

    print("[INFO:] Prepare Data")
    columns_string = NN_Overview.at[net_name, "feature"]
    prediction_columns_string = NN_Overview.at[net_name, "prediction"]

    columns = columns_string.split("'")[1:-1]
    columns = columns[::2]

    prediction_columns = prediction_columns_string.split("'")[1:-1]
    prediction_columns = prediction_columns[::2]

    w_timesteps = int(NN_Overview.at[net_name, "history_window"])
    p_timesteps = int(NN_Overview.at[net_name, "prediction_window"])
    batch_size = int(NN_Overview.at[net_name, "batch_size"])

    files = NN_Overview.at[net_name, "test_daten"]

    files = files.split("'")[1:-1]
    files = files[::2]

    return columns, prediction_columns,w_timesteps, p_timesteps, batch_size, files


def evaluate_net(net_name, model=None, csv_path=None, data_path=None, batch_size=None,
                 columns=None, prediction_columns=None, p_timesteps=None,
                 w_timesteps=None, boolean_norm_data=True, file_type=None, norm="ARENA",
                 file_length=86400, files=None, NN_Overview=None, NN_Overview_Name="NN_Overview.csv",
                 denorm=True, net_type='mxnet'):

    if model== None:
        file_path = csv_path + "LSTM_Nets/" + net_type + "/" + net_name + "/" + net_name + ".h5"

        print("[INFO:] Load Model: " + net_name)
        model = load_model(file_path=file_path)

    if NN_Overview == None:
        NN_Overview = load_Overview(csv_path,NN_Overview_Name)

    if columns == None:
        columns, prediction_columns, w_timesteps, p_timesteps, batch_size, files = \
            load_settings_form_Overview(NN_Overview, net_name)

    if file_type==None:
        file_type = "csv" # csv oder hdf

    if norm==None:
        norm = "ARENA"

    if denorm == None:
        denorm = True
        boolean_norm_data = True

    if net_type == "sinus":
        tune = 10
    else:
        tune = 1

    # baue DataGenerator
    gen = generator.generate_data_all_files(files=files, columns=columns,
                                            prediction_columns=prediction_columns, prediction_timeteps=p_timesteps,
                                            window_timeteps=w_timesteps, batch_size=batch_size, type="evaluiern",
                                            progress_tune=tune, LoadData=LoadData, data_path=data_path,
                                            csv_path=csv_path, norm=norm, normData=boolean_norm_data,
                                            filetype=file_type)

    # Vorbereitung der Daten zur De-Normierung der Werte
    if boolean_norm_data:
        if norm == "IEEE":
            maxValues = pd.read_csv(csv_path + "AbsMaxValues_IEEE.csv", index_col=False, header=0)
            minValues = pd.read_csv(csv_path + "AbsMinValues_IEEE.csv", index_col=False, header=0)
        elif norm == "sinus":
            maxValues = pd.read_csv(csv_path + "AbsMaxValues.csv", index_col=False, header=0)
            minValues = pd.read_csv(csv_path + "AbsMinValues.csv", index_col=False, header=0)
        else:
            maxValues = pd.read_csv(csv_path + "AbsMaxValues_IEEE_ARENA.csv", index_col=False, header=0)
            minValues = pd.read_csv(csv_path + "AbsMinValues_IEEE_ARENA.csv", index_col=False, header=0)

        minValues = minValues[prediction_columns]
        maxValues = maxValues[prediction_columns]

    names = list()
    for i in range(0, -p_timesteps - 1, -1):
        if i == 0:
            names += [(prediction_columns[j]) for j in range(len(prediction_columns))]
        else:
            names += [(prediction_columns[j] + '(t+%d)' % abs(i)) for j in range(len(prediction_columns))]

    prediction_DF = pd.DataFrame(data=[])
    real_DF = pd.DataFrame(data=[])

    prediction_norm_DF = pd.DataFrame(data=[])
    real_norm_DF = pd.DataFrame(data=[])

    print("[INFO:] Make Prediction")
    for i in range(0, len(files)*file_length, batch_size):
        # generiere TestDaten mit Datengenrator
        x, y = next(gen)

        real_norm_DF_series = pd.DataFrame(y)
        real_norm_DF = real_norm_DF.append(real_norm_DF_series, ignore_index=True)

        # Reale Daten in DataFrame
        if denorm:
            # denomiere TestDaten
            y = (y * (maxValues.loc[0].values - minValues.iloc[0].values)) + (minValues.iloc[0].values)

        real_DF_series = pd.DataFrame(y)
        real_DF = real_DF.append(real_DF_series, ignore_index=True)

        # erstelle eine Vorhersage
        prediction = model.predict(x, batch_size=batch_size, verbose=0, steps=None)

        predict_norm_DF_series = pd.DataFrame(prediction)
        prediction_norm_DF = prediction_norm_DF.append(predict_norm_DF_series, ignore_index=True)

        # Vorhersage Daten in DateFrame
        if denorm:
            # denomiere PrediktionsDaten übernehmen
            prediction = (prediction * (maxValues.loc[0].values - minValues.iloc[0].values)) + (minValues.iloc[0].values)

        predict_DF_series = pd.DataFrame(prediction)
        prediction_DF = prediction_DF.append(predict_DF_series, ignore_index=True)


    # METRICS Normiert !!!
    error_results = mt.Metrics(real_value=real_norm_DF, prognosis_value=prediction_norm_DF,
                               p_timesteps=p_timesteps)
    error_results.score_all()
    print(error_results.return_errors())

    # Metris abspeichern
    # mse_test	mae_test	mape_test
    NN_Overview.at[net_name, 'mse_test'] = error_results.mse
    NN_Overview.at[net_name, 'mae_test'] = error_results.mae
    NN_Overview.at[net_name, 'mape_test'] = error_results.mape
    NN_Overview.at[net_name, 'rmse_test'] = error_results.rmse

    # METRICS De-Normiert !!!
    denrom_error_results = mt.Metrics(real_value=real_DF, prognosis_value=prediction_DF,
                               p_timesteps=p_timesteps)
    denrom_error_results.score_linear_regression()
    print(denrom_error_results.return_denorm_error())
    NN_Overview.at[net_name, 'y_intercept'] = denrom_error_results.intercept
    NN_Overview.at[net_name, 'slope'] = denrom_error_results.slope
    NN_Overview.at[net_name, 'sigma'] = str(round(denrom_error_results.sigma, 40))[:1] \
                                         + str(round(denrom_error_results.sigma, 40))[-4:]
    NN_Overview.at[net_name, 'corrcoef'] = denrom_error_results.corrcoef
    NN_Overview.at[net_name, 'stand_err'] = str(round(denrom_error_results.stand_err, 40))[:1] \
                                             + str(round(denrom_error_results.stand_err, 40))[-4:]
    NN_Overview.at[net_name, 'r2'] = denrom_error_results.r2

    NN_Overview.to_csv(csv_path + NN_Overview_Name, float_format='%.5f')

    #real_DF = real_DF.transpose()
    history_path = csv_path + 'LSTM_Nets/' + net_type + '/' + net_name + '/'

    os.makedirs(history_path, exist_ok=True)

    print("[INFO:] Prediction is over")

    if args.plots == 1:
        plot = pt.Plot_Data(prediction_DF, real_DF, net_name, net_name, history_path,
                            s_plot=False, d_plot=True, unit=net_type)

        plot.prediction_real_progression(step=None, steprange=10, p_timesteps=p_timesteps)
        plot.prediction_real_tolerance(p_timesteps=p_timesteps)
        plot.prediction_area_error(p_timesteps=p_timesteps)
        plot.prediction_real_boxplot(step=1)
        plot.prediction_real_prozent_boxplot(step=1)

        plt.close("all")

    return real_DF, prediction_DF



def evaluate_multiple(n=10, NN_name="NN_Overview.csv", end=None, start=None, csv_path=None, data_path=None):


    NN_Overview = pd.read_csv(csv_path + NN_name)
    try:
        NN_Overview = NN_Overview.drop('Unnamed: 0', axis=1)
    except KeyError:
        pass

    NN_Overview = NN_Overview.set_index("name")
    if start!=None:
        for name in NN_Overview.index[start:end]:
            print(name)
            evaluate_net(name, csv_path=csv_path, data_path=data_path,
                         boolean_norm_data=True, norm="ARENA",
                         denorm=True)
    else:
        for name in NN_Overview.index[-n:]:
            print(name)
            evaluate_net(name, csv_path=csv_path, data_path=data_path, batch_size=None,
                         prediction_columns=None, w_timesteps=None, boolean_norm_data=True,
                         norm="ARENA", file_length=24, denorm=True)


def evaluate_sinus(name):
    data_path = "/share/Shared/SinusData/sinustestdaten/"
    csv_path = "/share/Shared/masterarbeit/Share/Code/src/Felix/Data/"
    evaluate_net(name, csv_path=csv_path, data_path=data_path, boolean_norm_data=True,
                 file_type="csv", norm="sinus", file_length=200, denorm=True,
                 NN_Overview_Name="NN_SinusError_new.csv", net_type="sinus")


def evaluate_current(name):
    data_path = "/share/Shared/DATEN_AUS_ARENA/Trafo1/h5/"
    csv_path = "/share/Shared/masterarbeit/Share/Code/src/Felix/Data/"

    evaluate_net(name, csv_path=csv_path, data_path=data_path, boolean_norm_data=True,
                 denorm=True, net_type="current",
                 NN_Overview_Name="NN_rms_i1_Error.csv")

def evaluate_voltage(name):
    data_path = "/share/Shared/DATEN_AUS_ARENA/Trafo1/h5/"
    csv_path = "/share/Shared/masterarbeit/Share/Code/src/Felix/Data/"

    evaluate_net(name, csv_path=csv_path, data_path=data_path, boolean_norm_data=True,
                 denorm=True, net_type="voltage",
                 NN_Overview_Name="NN_rms_u1_Error.csv")


if __name__ == "__main__":
    args = parser.parse_args()

    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    #csv_path = "/share/Shared/masterarbeit/Share/Code/src/Felix/Data/"
    csv_path = "/Users/Array/PycharmProjects/Bachelor/ANN_Daten/"

    print(args.evalu_type)

    if args.evalu_type=="sinus":
        if (args.name == ""):
            data_path = '/share/Shared/SinusData/'
            files = os.listdir(csv_path + 'LSTM_Nets/sinus/')
            try:
                files.remove('.DS_Store')
            except ValueError:
                pass
            for i in files:
                try:
                    evaluate_sinus(i)
                except KeyError:
                    pass
        else:
            evaluate_sinus(args.name)

    elif args.evalu_type=="strom":
        if args.name == "":
            files = os.listdir(csv_path + 'LSTM_Nets/current/')
            try:
                files.remove('.DS_Store')
            except ValueError:
                pass

            NN_Overview = pd.read_csv(csv_path + "NN_rms_i1_Error.csv")
            try:
                NN_Overview = NN_Overview.drop('Unnamed: 0', axis=1)
            except KeyError:
                pass

            NN_Overview = NN_Overview.set_index("name")[:]
            if args.net_name != None:
                print(args.net_name)
                evaluate_current(args.net_name)
            else:
                for i in NN_Overview.index[args.sum_evatuation:]:
                    try:
                        evaluate_current(i)
                    except KeyError:
                        pass

    elif args.evalu_type == "spannung":
        if args.name == "":

            files = os.listdir(csv_path + 'LSTM_Nets/voltage/')
            try:
                files.remove('.DS_Store')
            except ValueError:
                pass
            NN_Overview = pd.read_csv(csv_path + "NN_rms_u1_Error.csv")
            try:
                NN_Overview = NN_Overview.drop('Unnamed: 0', axis=1)
            except KeyError:
                pass

            NN_Overview = NN_Overview.set_index("name")
            for i in NN_Overview.index[args.sum_evatuation:]:
                try:
                    print(i)
                    evaluate_voltage(i)
                except KeyError:
                    pass

    elif args.evalu_type == "raw":
        if args.name == "":

            # files = os.listdir(csv_path + 'LSTM_Nets/mxnet/')
            # try:
            #     files.remove('.DS_Store')
            # except ValueError:
            #     pass
            NN_Overview = pd.read_csv(csv_path + "NN_Overview.csv")
            try:
                NN_Overview = NN_Overview.drop('Unnamed: 0', axis=1)
            except KeyError:
                pass

            NN_Overview = NN_Overview.set_index("name")
            for i in NN_Overview.index[args.sum_evatuation:]:
                try:
                    print(i)
                    #data_path = "/share/Shared/DATEN_AUS_ARENA/Trafo1/h5/"
                    #csv_path = "/share/Shared/masterarbeit/Share/Code/src/Felix/Data/"

                    csv_path = "/Users/Array/PycharmProjects/Bachelor/ANN_Daten/"
                    data_path = "/Users/Array/PycharmProjects/Bachelor/Daten/20180720_E_TR1_1_DftL1C02/"

                    # evaluate_net(i, csv_path=csv_path, data_path=data_path, boolean_norm_data=False,
                    #              denorm=False, net_type="mxnet",
                    #              NN_Overview_Name="NN_Overview.csv")
                    evaluate_net(i, csv_path=csv_path, data_path=data_path, boolean_norm_data=False,
                                 denorm=False, net_type="tensorflow",
                                 NN_Overview_Name="NN_Overview.csv")

                except KeyError:
                    pass


                #python3 /Users/Array/PycharmProjects/Bachelor/LSTM/EvaluateNet.py -T raw
