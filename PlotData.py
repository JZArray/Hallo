#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 09 11:25:19 2018

@author: felix
"""

import numpy as np
import pandas as pd
from scipy import stats
import calendar
import matplotlib.dates as mdates
import datetime as dt
import matplotlib as mpl
mpl.use('agg')
import matplotlib.dates as pltdt
import datetime as dt
import matplotlib.pyplot as plt
import math
import os

from matplotlib import pyplot as plt
from load_data import LoadData as LoadData


class Plot_Data:
    def __init__(self,
                 prediction_df,  # Prognose der Testdaten
                 original_df,  # Test Outputdaten
                 name,  # Bezeeichung des Datensatzes
                 filename,  # Bezeeichung des Datensatzes
                 path_name,  # Dateipfad
                 s_plot=True,  # show plots
                 d_plot=False, # save plots
                 min_max=True,
                 unit=None):

        if (min_max):
            if prediction_df.values.max() < original_df.values.max():
                self.max_range = original_df.values.max()
            else:
                self.max_range = prediction_df.values.max()

            if prediction_df.values.min() > original_df.values.min():
                self.min_range = original_df.values.min()
            else:
                self.min_range = prediction_df.values.min()

        self.prediction_df = prediction_df
        self.original_df = original_df
        self.error_matrix = self.original_df - self.prediction_df
        self.error_matrix_p = (self.error_matrix / self.original_df) * 100
        self.name = name
        self.filename = filename

        self.s_plot = s_plot
        self.d_plot = d_plot

        if unit == "current":
            self.label_name = 'Strom in Ampere'
        elif unit == "voltage":
            self.label_name = 'Spannung in Volt'
        elif unit == "sinus":
            self.label_name = ''
        else:
            self.label_name = ''

        self.path_name = path_name

        # Farben
        self.c1 = 'royalblue'
        self.c2 = 'darkcyan'
        self.c3 = 'mediumslateblue'
        self.n1 = 'firebrick'
        self.n2 = 'coral'
        self.n3 = 'palevioletred'
        self.a1 = 'ligthcoral'
        self.a2 = 'orange'
        self.a3 = 'yellowgreen'
        self.a4 = 'mediumturquoise'
        self.a5 = 'dodgerblue'
        self.a6 = 'slateblue'
        self.a7 = 'violet'
        self.a8 = 'crimson'

        self.color_range = ['royalblue', 'darkcyan', 'mediumslateblue', 'firebrick', 'coral', 'palevioletred',
                            'dodgerblue', 'slateblue', 'royalblue', 'mediumslateblue','indigo','aliceblue','mediumpurple',
                            'darkkhaki','darkgrey','lightcoral','lawngreen','seashell','thistle','antiquewhite']

        self.s_size = 12
        self.m_size = 14
        self.b_size = 16

        self.f_size = (13, 6.5)
        self.l_loc = 'center left'
        self.l_bbox = (1, 0.5)
        self.adj = 0.75

        plt.rc('font', size=self.s_size)  # controls default text sizes
        plt.rc('axes', titlesize=self.b_size)  # fontsize of the axes title
        plt.rc('axes', labelsize=self.m_size)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=self.m_size)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=self.m_size)  # fontsize of the tick labels
        #        plt.rc('legend', fontsize=self.m_size)    # legend fontsize
        plt.rc('figure', titlesize=self.b_size)  # fontsize of the figure title

        #self.split_name()

    # ==============================================================================
    def savefig(self, plot_name):

        if self.d_plot:
            plt.savefig(self.path_name + plot_name + '.png', dpi=300)

    def showfig(self, plot_name):

        if self.s_plot:
            plt.show(plot_name)
        else:
            plt.close(plot_name)

    def split_name(self):
        first = self.data_name.split("l")
        layer = first[1].split("n")
        neurons = layer[1].split("e")
        fourth = neurons[1].split("bs")

        self.name = self.data_name + "\n mit: " + layer[0] +" Schichten, "+ neurons[0] +" Neuronen, "+ \
                    fourth[0] + " Epochen, " + fourth[1] + " Batchgröße, "

    def prediction_real_progression(self, step, steprange, p_timesteps):
        print("[INFO:] plot progress")
        fig = plt.figure(figsize=self.f_size)

        if step == None:
            x = range(0, steprange * p_timesteps, p_timesteps)
            x2= range(0, steprange * p_timesteps)
            plt.plot(x2, self.original_df.values[0:steprange*p_timesteps, 0], self.c2, label="real")
            for i in x:
                x1 = range(i, i + p_timesteps)
                plt.plot(x1, self.prediction_df.values[i], self.c1, label="prediction_"+str(i))

        else:
            plt.plot(self.original_df.values[step], self.c2, label="real")
            plt.plot(self.prediction_df.values[step], self.c1, label="prediction")

        plt.grid(linestyle='-', linewidth=0.5)
        plt.xlabel('Zeitschritte in Sekunden')
        plt.ylabel(self.label_name)

        plt.subplots_adjust(right=self.adj)
        plt.legend(fontsize=self.m_size, loc=self.l_loc, bbox_to_anchor=self.l_bbox)
        plt.title('Verlauf der Vorhersage mit Netz ' + self.name +'\n')

        self.savefig(self.name + "_progression")
        self.showfig(self.filename + " Progression")

    def real_progression(self, lables, unit, offset=3):
        print("[INFO:] plot progress")
        fig = plt.figure(figsize=self.f_size)

        for c in range(0, len(lables)):
            plt.plot(self.original_df[lables[c]], color=self.color_range[c+3], linewidth=0.3)

        plt.grid(linestyle='-', linewidth=0.5)
        plt.xlabel('Tage des Monats Mai \n')
        plt.ylabel(self.label_name)

        plt.subplots_adjust(right=self.adj)
        plt.legend(fontsize=self.m_size, loc=self.l_loc, bbox_to_anchor=self.l_bbox)
        plt.title('Verlauf der ' + self.label_name+ ' im Monat Mai')

        day_length = 86400

        plt.xticks([0* day_length, 7 * day_length, 14* day_length, 21* day_length,  28* day_length],
                   ["1. Mai", "7. Mai", "12. Mai", "21. Mai", "28. Mai"])

        self.savefig(self.name + "_progression")
        self.showfig(self.filename + " Progression")

    def prediction_real_tolerance(self, p_timesteps, step=1):
        print("[INFO:] plot Scatter")
        # Scatter plot
        fig, ax = plt.subplots(figsize=self.f_size)

        x = np.arange(int(self.min_range), int(self.max_range))

        a, b = [], []
        for s in range(0, p_timesteps):
            a1 = self.prediction_df.values[:, s]
            b1 = self.original_df.values[:, s]

            a = np.concatenate([a,a1])
            b = np.concatenate([b, b1])

        slope, intercept, r_value, p_value, std_err = stats.linregress(b, a)

        print("r2: " + str(r_value**2))

        z1 = np.polyfit(b, a, deg=1)
        p1 = np.poly1d(z1)
        print(str(p1)[2:])

        plt.plot(x, p1(x), c="k", linestyle="-", label='Lineare Regression')

        sigma = str(round(std_err**2,40))

        text = r'$y  = $' + str(p1)[2:] + "\n" + r'$r^2 = $' +\
               str(round(r_value**2, 4)) + "\n" + r'$\sigma^2 = $' + \
               sigma[:6]+sigma[-4:]

        if p_timesteps < 5:
            alpha = 0.7
            alpha_m = 0.1

        elif p_timesteps < 10:
            alpha = 1
            alpha_m = 0.07

        elif p_timesteps < 15:
            alpha = 1
            alpha_m = 0.04

        else:
            alpha = 1
            alpha_m = 0.02

        c = 0
        for i in range(0, p_timesteps, step):
            ax.scatter(self.original_df.values[:, i], self.prediction_df.values[:, i], s=10, label='Vorhersagewert '
                                                                                                   + str(i+1),
                       color=self.color_range[c], alpha=alpha)
            alpha -= alpha_m
            c += 1

        plt.ylabel('Vorhersagewert ' + self.label_name)
        plt.xlabel('Messwert ' + self.label_name)
        plt.title('Messwert vs Vorhersage für Netz ' + self.name+'\n')

        axes = plt.gca()

        axes.set_ylim([self.min_range, self.max_range])
        axes.set_xlim([self.min_range, self.max_range])
        ax.grid(True)

        if self.name.find("sinus"):

            plt.text(self.max_range - self.max_range * 0.05,
                     self.min_range + self.min_range * 0.05, text
                     , fontsize=12)
        else:
            plt.text(self.max_range - self.max_range * 0.8,
                     self.min_range + self.min_range * 0.2, text
                     , fontsize=12)

        plt.subplots_adjust(right=self.adj)
        plt.legend(fontsize=self.m_size, loc=self.l_loc, bbox_to_anchor=self.l_bbox)

        self.savefig(self.name + "_tolerance")
        self.showfig(self.filename + " Tolerance")


    def prediction_area_error(self, p_timesteps, step=1):
        print("[INFO:] plot Scatter II")
        # Scatter plot
        fig, ax = plt.subplots(figsize=self.f_size)

        if p_timesteps < 5:
            alpha = 0.7
            alpha_m = 0.1

        elif p_timesteps < 10:
            alpha = 1
            alpha_m = 0.07

        elif p_timesteps < 15:
            alpha = 1
            alpha_m = 0.04

        else:
            alpha = 1
            alpha_m = 0.02

        c = 0
        for i in range(0, p_timesteps, step):
            ax.scatter(self.original_df.values[:, i], self.error_matrix.values[:, i], s=10, label='Vorhersagewert '
                                                                                                   + str(i+1),
                       color=self.color_range[c], alpha=alpha)
            alpha -= alpha_m
            c += 1

        plt.ylabel('Abweichung ' + self.label_name)
        plt.xlabel('Messwert ' + self.label_name)
        plt.title('Fehler je Messwertebereich für Netz ' + self.name+'\n')

        ax.grid(True)

        plt.subplots_adjust(right=self.adj)
        plt.legend(fontsize=self.m_size, loc=self.l_loc, bbox_to_anchor=self.l_bbox)

        self.savefig(self.name + "_error_area")
        self.showfig(self.filename + " Error Area")

    def prediction_real_boxplot(self, step=10, steprange=None, unit="I"):
        print("[INFO:] plot Box")

        fig, ax = plt.subplots(figsize=self.f_size)

        ax.boxplot(self.error_matrix.values[:, ::step])

        plt.ylabel('Abweichung ' + self.label_name)

        labels = [item.get_text() for item in ax.get_xticklabels()]
        for l in range(len(labels)):
            labels[l] = l*step

        ax.set_xticklabels(labels)

        plt.xlabel('Prognose Zeitschritte')
        plt.title('Boxplot für Netz ' + self.name)

        ax.grid(True)

        self.savefig(self.name + "_boxplot")
        self.showfig(self.filename + " Tolerance")

    def prediction_real_prozent_boxplot(self, step=10, steprange=None, unit="I"):
        print("[INFO:] plot Box")

        fig, ax = plt.subplots(figsize=self.f_size)

        ax.boxplot(self.error_matrix_p.values[:, ::step])

        plt.ylabel('Abweichung Prozent')

        labels = [item.get_text() for item in ax.get_xticklabels()]
        for l in range(len(labels)):
            labels[l] = l*step

        ax.set_xticklabels(labels)

        plt.xlabel('Prognose Zeitschritte')
        plt.title('Boxplot für Netz ' + self.name)

        ax.grid(True)

        self.savefig(self.name + "_boxplot_prozent")
        self.showfig(self.filename + " Tolerance")

    def prediction_real_historgam(self, step=10, steprange=None, unit="I"):
        print("[INFO:] plot Histogram")

        fig, ax = plt.subplots(figsize=self.f_size)

        ax.hist(self.error_matrix.values[:,0], 1000)

        plt.ylabel('Anzahl')

        plt.xlabel('Abweichung in ' + self.label_name)
        plt.title('Histogram für Netz ' + self.name)

        ax.grid(True)

        self.savefig(self.name + "_histogram")
        self.showfig(self.filename + " Tolerance")
