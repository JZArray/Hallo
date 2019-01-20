#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 11:54:29 2018

@author: felix
"""

import pandas as pd
import os
import datetime as dt


class LoadData:

    def __init__(self, data_path, filename, csv_path, fileType="csv", norm="", BooleanNormValues=True):
        # liest die Daten ein und prüft welcher Filetype verlangt wird
        self.shiftedData = None
        self.normData = None
        self.arrayData = None
        self.BooleanNormValues = BooleanNormValues

        if fileType == "csv":
            self.data = pd.read_csv(data_path + filename)
        elif fileType == "hdf":
            self.data = pd.read_hdf(data_path + filename)
        else:
            print("Please check your file type.")
        #self.matrix = self.data.as_matrix()
        if BooleanNormValues:
            if norm == "IEEE":
                self.maxValues = pd.read_csv(csv_path + "AbsMaxValues_IEEE.csv", index_col=False, header=0)
                self.minValues = pd.read_csv(csv_path + "AbsMinValues_IEEE.csv", index_col=False, header=0)
            elif norm == "entsoe":
                self.maxValues = pd.read_csv(csv_path + "Abs_Max_Values_Entsoe.csv", index_col=False, header=0)
                self.minValues = pd.read_csv(csv_path + "Abs_Min_Values_Entsoe.csv", index_col=False, header=0)
            elif norm == "ARENA":
                self.maxValues = pd.read_csv(csv_path + "AbsMaxValues_IEEE_ARENA.csv", index_col=False, header=0)
                self.minValues = pd.read_csv(csv_path + "AbsMinValues_IEEE_ARENA.csv", index_col=False, header=0)
            else:
                self.maxValues = pd.read_csv(csv_path+"AbsMaxValues.csv", index_col=False, header=0)
                self.minValues = pd.read_csv(csv_path+"AbsMinValues.csv", index_col=False, header=0)

            self.dataRange = self.maxValues.loc[0] - self.minValues.iloc[0]

            try:
                self.maxValues.drop(["Unnamed: 0"], axis=1, inplace=True)
                self.minValues.drop(["Unnamed: 0"], axis=1, inplace=True)
            except KeyError:
                pass

        #list containing all features (with t+...), constructed at shiftData
        self.feature_names = None


        #self.maxValues.columns, self.minValues.columns = ["names", "values"]


    def shiftData(self, shiftCount=1, dropnan=True, forecast=True):
        """In der Shift Data Methode werden die Daten für eine mögliche Prognose vorbereitet
        und um einen gewünschten Faktor in die Zukungt geschoben. Die Methode dient der Erstellung
        von Trainingsdaten."""
        names, columns = list(), list()

        if forecast:
            sign = "+"
            forrange = range(0, -shiftCount - 1, -1)
        else:
            sign = "-"
            forrange = range(0, shiftCount + 1)

        for i in forrange:
            columns.append(self.data.shift(i))
            if i == 0:
                names += [(self.data.columns[j]) for j in range(len(self.data.columns))]
            else:
                names += [(self.data.columns[j] + '(t' + sign + '%d)' % abs(i)) for j in range(len(self.data.columns))]

        aggregated = pd.concat(columns, axis=1)
        aggregated.columns = names
        self.feature_names = names

        if dropnan:
            aggregated.dropna(inplace=True)

        self.shiftedData = aggregated

    def normValues(self, csvPath="Stromtankstelle/"):
        """Die Methoder normiert die Werte auf einen Wert zwischen 0 und 1.
        Die Zugrunde liegende Formerl lautet: (x - min)/(max - min)"""
        self.data = (self.data.sub(self.minValues.iloc[0], axis=1))
        self.data = self.data.div((self.maxValues.loc[0] - self.minValues.iloc[0]), axis=1)

    def selectParams(self, columns):
        """Verkürtz den Datensatz"""
        self.data = self.data.loc[:,columns]
        if self.BooleanNormValues:
            self.minValues = self.minValues[columns]
            self.maxValues = self.maxValues[columns]

    def getData(self):
        # Gibt die Daten als DataFrame zurück
        return self.data

    def getShiftedData(self):
        # Gibt die Daten als DataFrame zurück
        return self.shiftedData

    def getNormData(self):
        # Gibt die Daten als DataFrame zurück
        return self.normData

    def getMaxData(self):
        # Gibt die Daten als DataFrame zurück
        return self.maxValues

    def getMinData(self):
        # Gibt die Daten als DataFrame zurück
        return self.minValues


'''#####################################################################################################'''


# def findMinMax():
#     x = 0
#     data_path = "/media/ssd2/20180220_PQ_Data_Stromtankstelle_Stoeckach/allpqdata/"
#     files = os.listdir(data_path)
#     columnNames = ['N0', 'N1', 'N2', 'N3', 'P0', 'P1', 'P2', 'P3', 'PF0', 'PF1', 'PF2', 'PF3',
#                    'S0', 'S1', 'S2', 'S3', 'freq0', 'freq1', 'freq2', 'freq3',
#                    'rms_i0', 'rms_i1', 'rms_i2', 'rms_i3', 'rms_u0', 'rms_u1', 'rms_u2', 'rms_u3',
#                    'thd_u0', 'thd_u1', 'thd_u2', 'thd_u3']
#
#     preResult, result = pd.DataFrame(), pd.DataFrame()
#     maxValues, minValues = pd.DataFrame(), pd.DataFrame()
#
#     min_names, max_names = list(), list()
#
#     for col in columnNames:
#         min_names += [col + "_min"]
#         max_names += [col + "_max"]
#
#     for file in files:
#         print(file)
#         date = dt.datetime.strptime(file[:-3], "%Y-%m-%d")
#         data = pd.read_hdf(data_path + file)
#         max_df = pd.DataFrame([data[columnNames].max().values], columns=max_names)
#         min_df = pd.DataFrame([data[columnNames].min().values], columns=min_names)
#
#         preResult = pd.concat([max_df, min_df], axis=1)
#
#         preResult["filename"] = file
#         preResult["date"] = date
#
#         result = result.append(preResult, ignore_index=True)  # , axis= 0, )
#
#         maxValues = maxValues.append([data[columnNames].max()], ignore_index=True)
#         minValues = minValues.append([data[columnNames].min()], ignore_index=True)
#
#     # result.to_csv("MinMaxValues.csv")
#     absMaxValues = maxValues.max()
#     absMinValues = minValues.min()
#
#     return result, absMaxValues, absMinValues
#
#
#
# def appendAll():
#     x = 0
#     data_path = "/media/ssd2/20180220_PQ_Data_Stromtankstelle_Stoeckach/allpqdata/"
#     files = os.listdir(data_path)
#     for filename in files:
#
#         data = pd.read_hdf(data_path + filename)
#         '''row = pd.DataFrame([[dt.datetime.strptime(filename[:-3], "%Y-%m-%d"),
#                             filename,len(data.getData().columns)]],
#                            columns = ["Date","FileName","Length"])
#         result = result.append(row)'''
#         print(x, " von ", len(files))
#         if x == 0:
#             if len(data.columns) == 36:
#                 prepared = data.drop(['thd_i0', 'thd_i1', 'thd_i2', 'thd_i3'], axis=1)
#                 result = pd.DataFrame(prepared)
#             else:
#                 result = pd.DataFrame(data.values)
#         else:
#             if len(data.columns) == 36:
#                 prepared = data.drop(['thd_i0', 'thd_i1', 'thd_i2', 'thd_i3'], axis=1)
#                 result = pd.concat([result, prepared], axis=0)
#             else:
#                 result = pd.concat([result, data], axis=0)
#         x += 1
#     return result
#
#
# if __name__ == '__main__':
#     data_path = "/media/ssd2/20180220_PQ_Data_Stromtankstelle_Stoeckach/allpqdata/"
#     csv_path = "/home/fst/masterarbeit/Share/Code/src/Felix/Data/"
#     files = os.listdir(data_path)
#     filename = files[0]
#     data = LoadData(data_path=data_path, filename=filename, csv_path=csv_path, norm="IEEE")
#     pass
