import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from __future__ import division
from matplotlib import colors as mcolors
from robust_synth import *


class Synth():
    def __init__(self, treat_unit, year, eta=0, p=1):
        self.treat_unit = treat_unit
        self.year = year
        self.eta = eta
        self.p = p
        self.w = []
        self.estimate = []
        self.raw = []

    def fit(self, df, num_sv=1, method="linear", method2="row avg", drop=0, drop_list=[]):
        data = df.copy()
        self.num_sv = num_sv

        # prepare data
        self.drop = drop
        if self.drop == 1:
            data = data.drop(data.index[drop_list])
        X = data.as_matrix()
        donor_list = list(data.index)

        # treated unit
        unit = donor_list.index(self.treat_unit)

        # let row one be treated unit
        X = swap(X, unit)
        self.raw = np.copy(X[0, :])
        self.Y = np.copy(X)

        # drop p proportion of entries
        idx1, idx2 = MAR(X, self.p)
        X = impute(X, idx1, idx2, method=method2)

        # estimation
        self.method = method
        self.w, self.estimate, self.M_hat, self.sigma_hat = learning(
            X, self.year, num_sv=self.num_sv, method=self.method, eta=self.eta)

    def vis_data(self, xlabel, ylabel, title, year_shift, year_mod=5,
                 legend_loc="upper left", line_width=2.0, frame_color='0.925'):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.year_shift = year_shift
        self.year_mod = year_mod
        self.legend_loc = legend_loc
        self.line_width = line_width
        self.frame_color = frame_color

    def vis(self):
        self.estimate_label = "Synthetic " + self.treat_unit
        self.raw_label = self.treat_unit
        if self.drop == 1:
            visuals(self.estimate, self.raw, self.estimate_label, self.raw_label, self.year, self.year_shift,
                    self.xlabel, self.ylabel, self.title + " - no bad states", legend_loc=self.legend_loc,
                    year_mod=self.year_mod, line_width=self.line_width, frame_color=self.frame_color)
        else:
            visuals(self.estimate, self.raw, self.estimate_label, self.raw_label, self.year, self.year_shift,
                    self.xlabel, self.ylabel, self.title, legend_loc=self.legend_loc,
                    year_mod=self.year_mod, line_width=self.line_width, frame_color=self.frame_color)


# BAYESIAN METHOD
""" BASQUE STUDY """
df = pd.read_csv('basque.csv')
data = df.pivot_table(values='gdpcap', index='regionname', columns='year')
data = data.drop('Spain (Espana)')

# Analyze economy of Basque country
year = 15
year_shift = 1955

for n in range(1, 8):
    # Basque country (ols)
    region = "Basque Country (Pais Vasco)"
    ols_ = Synth(region, year)
    ols_.fit(data, num_sv=n, method="Bayes")
    #ols_.vis_data(xlabel=xlabel, ylabel=ylabel, title=title, year_shift= year_shift, legend_loc=legend_loc)
    # ols_.vis()
    ols = ols_.estimate

    year = 15
    year_shift = 1955
    xlabel = "year"
    ylabel = "real per-capita GDP (1986 USD, thoudsand)"
    title = "Basque country study: # singular values = " + str(n)

    fig, ax = plt.subplots()
    ax.plot(ols_.raw, label='Basque Country', linewidth=1.75, color='k')
    ax.plot(ols_.estimate, '--', label='Robust synthetic Basque (Bayes)', linewidth=1.75, color='b')
    ax.plot(abadie_y, '--', label='Original synthetic Basque', linewidth=1.75, color='g')

    x = np.linspace(0, 42, 43)
    clr1 = 'lightcyan'
    clr2 = 'paleturquoise'
    ax.fill_between(x, ols_.estimate, ols_.estimate + ols_.sigma_hat,
                    facecolor=clr1, edgecolor=clr2, interpolate=True)
    ax.fill_between(x, ols_.estimate, ols_.estimate - ols_.sigma_hat,
                    facecolor=clr1, edgecolor=clr2, interpolate=True)

    legend = ax.legend(loc='lower right', shadow=True, prop={'size': 9.5})
    frame = legend.get_frame()
    frame.set_facecolor('0.925')
    ax.plot([year, year], [0, 18], '--', linewidth=1.5, color='r')
    years = int(np.floor(ols_.raw.shape[0] / 5))
    x = np.array([5 * i for i in range(years + 1)])

    ax.set_ylim([0, 18])

    plt.xticks(x, x + year_shift)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
