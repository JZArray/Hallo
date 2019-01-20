#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 16:50:29 2018

@author: felix
"""
import numpy as np
import os
#from load_data import LoadData as LoadData

def generate_data_all_files(files, columns, prediction_columns, prediction_timeteps, window_timeteps, batch_size,
                            type, progress_tune, LoadData, data_path, csv_path, norm="IEEE", normData=False,
                            filetype="csv", verbose=0):
    while True:
        progress = 0
        for filename in files:
            if verbose == 1:
                # soll der Progress gezeit werden oder nicht
                if progress % progress_tune == 0:
                    progress_bar(progress/len(files)*100, type)

                progress += 1

                if progress == len(files):
                    progress = 0

            data = LoadData(data_path=data_path, filename=filename, csv_path=csv_path, norm=norm, fileType=filetype,
                        BooleanNormValues=normData)

            # Wählt nur bestimmte Werte der Datenreihe aus
            data.selectParams(columns)


            # der wert für das Zeitfenster bezieht sich auf die gesamte Reihe also auch den Wert t=0
            # so muss der Wert um 1 reduziert werden

            if (normData):
                # Normiert alle Daten
                data.normValues()

            # Läd Feature-Array(X) mit Dimensionen:
            # (Samples, Features)
            train_x_xy = data.getData()
            # print('train_x_xy:')
            # print(train_x_xy.shape)
            # print(train_x_xy)
            # print(train_x_xy.values)



            # Läd Feature-Array(X) mit Dimensionen:
            # (Samples, Window_Timesteps)
            if window_timeteps > 1:
                data.shiftData(window_timeteps-1, forecast=False)
                train_x_z = data.getShiftedData()
                train_x_z = train_x_z.drop(columns, axis=1)
                # print('train_x_z:')
                # print(train_x_z.shape)

            # Wält Daten aus die Prognostiziert werden sollen aus und verschiebt diese in die
            # Zukunft und extrahiert daraus die Label
            data.selectParams(prediction_columns)
            data.shiftData(prediction_timeteps, forecast=True)
            train_y = data.getShiftedData()
            # print('train_y: ohne Drop')
            #print(train_y.index[-1])
            #print(train_y.shape)
            train_y = train_y.drop(prediction_columns, axis=1)
            # print('train_y: nach Drop')
            # print(train_y.shape)

            # Im folgenden Schritt werden die Daten auf die gleiche Länge gebracht und
            # in Arrays gewandelt
            if window_timeteps == 1:
                train_x_xy = train_x_xy.loc[:train_y.index[-1]]

            else:
                train_x_xy = train_x_xy.loc[train_x_z.index[0]:train_y.index[-1]]#.loc的区间为闭区间 所以要取最后一行的index 而不是全部行数
                #train_x_xy = train_x_xy.loc[train_x_z.index[0]:train_y.shape[0] - 1]
                #print('train_x_xy: 合并之后')
                #print(train_x_xy.shape)
                # print(train_x_xy)
                train_x_z = train_x_z.loc[:train_y.index[-1]]
                #print('train_x_z: 合并之后')
                #print(train_x_z.shape)
                train_y = train_y.loc[train_x_z.index[0]:]
                #print('train_y: 合并之后')
                #print(train_y.shape)

            train_x_xy = train_x_xy.values
            # print('train_x_xy转型')
            # print(train_x_xy.shape)
            # print(train_x_xy)

            # Im Reshape wird das Feature-Array(X) in die Dimensionen:
            # (Samples, Timesteps, Features) gebracht
            if window_timeteps > 1:
                for t in range(0, (window_timeteps-1)*len(columns), len(columns)):
                    if t == 0:
                        print(train_x_z.iloc[:, t:t+len(columns)].values.shape)
                        train_x = np.stack([train_x_xy, train_x_z.iloc[:, t:t+len(columns)].values], axis=1)
                        #print("t = 0 时的train_x:")
                        #print(train_x.shape)
                        #print(train_x)
                    else:
                        appendix = train_x_z.iloc[:, t:t+len(columns)].values
                        appendix = appendix.reshape((appendix.shape[0], 1, appendix.shape[1]))
                        train_x = np.append(train_x, appendix, axis=1)
                        #print('最后的train_x:')
                        #print(train_x.shape)
            else:
                train_x = train_x_xy
                train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))

            # Extract Arrays
            train_y = train_y.values
            #train_y = train_y.reshape((train_y.shape[0], 1, train_y.shape[1]))

            #print(train_x.shape[0])
            for i in range(0, train_x.shape[0], batch_size):
                if train_x[i:i+batch_size].shape[0] == batch_size & train_y[i:i+batch_size].shape[0]:
                    yield train_x[i:i+batch_size], train_y[i:i+batch_size]
                else:
                    pass

def progress_bar(progress, info):
    print(info + " " + "#" * int(progress / 2) + "-" * (50 - int(progress / 2)) + " : " +
          str(round(progress, 2)) + "%")

