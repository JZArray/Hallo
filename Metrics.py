#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 17:55:11 2018

@author: felix
"""
from math import sqrt
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Metrics:
    def __init__(self, real_value, prognosis_value, p_timesteps):
        self.real_value = real_value
        self.prognosis_value = prognosis_value
        self.p_timesteps = p_timesteps

        self.mae = None
        self.mse = None
        self.mape = None
        self.rmse = None

        self.r2 = None
        self.sigma = None
        self.slope = None
        self.intercept = None
        self.stand_err = None
        self.corrcoef = None

        self.error_matrix = None

    def score_mae(self):
        self.mae = mean_absolute_error(self.real_value, self.prognosis_value)

    def score_mse(self):
        self.mse = mean_squared_error(self.real_value, self.prognosis_value)

    def score_mape(self):
        self.mape = np.mean(np.mean(np.abs((self.real_value - self.prognosis_value) / self.real_value))) * 100

    def score_rmse(self):
        self.rmse = sqrt(mean_squared_error(self.real_value, self.prognosis_value))

    def score_linear_regression(self):
        a, b = [], []
        for s in range(0, self.p_timesteps):
            a1 = self.prognosis_value.values[:, s]
            b1 = self.real_value.values[:, s]

            a = np.concatenate([a, a1])
            b = np.concatenate([b, b1])

        slope, intercept, r_value, p_value, std_err = stats.linregress(b, a)

        self.slope = slope
        self.r2 = r_value**2
        self.intercept = intercept
        self.corrcoef = r_value
        self.stand_err = std_err
        self.sigma = std_err ** 2

    def score_all(self):
        self.score_mae()
        self.score_mse()
        self.score_mape()
        self.score_rmse()

    def score_error_matrix(self):
        self.error_matrix = self.real_value - self.prognosis_value

    def return_errors(self):
        return self.mae, self.mse, self.mape, self.rmse

    def return_denorm_error(self):
        return self.sigma, self.slope, self.intercept, self.stand_err, self.corrcoef, self.r2
