from __future__ import division
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from robust_synth import *


# BAYESIAN METHOD
""" BASQUE STUDY """
df = pd.read_csv('basque.csv')
data = df.pivot_table(values='gdpcap', index='regionname', columns='year')
data = data.drop('Spain (Espana)')

# Analyze economy of Basque country
year = 15
year_shift = 1955
region = "Basque Country (Pais Vasco)"
basque = Synth(region, year=year, method="Bayes")
basque.fit(data, num_sv=2)

xlabel = "year"
ylabel = "per-capita gdp"
title = "Basque Country Case Study"

# plot results
fig, ax = plt.subplots()
ax.plot(basque.raw, label="Basque Country", linewidth=1.75, color='g')
ax.plot(basque.estimate, label="Robust synthetic Basque", linewidth=1.75, color='b', --)
x = np.linspace(0, 42, 43)
clr1 = 'lightcyan'
clr2 = 'paleturquoise'
ax.fill_between(x, basque.estimate, basque.estimate + basque.sigma_hat,
                facecolor=clr1, edgecolor=clr2, interpolate=True)
ax.fill_between(x, basque.estimate, basque.estimate - basque.sigma_hat,
                facecolor=clr1, edgecolor=clr2, interpolate=True)

legend = ax.legend(loc='lower right', shadow=True, prop={'size': 9.5})
frame = legend.get_frame()
frame.set_facecolor('0.925')
ax.plot([year, year], [0, 18], '--', linewidth=1.5, color='r')
years = int(np.floor(basque.raw.shape[0] / 5))
x = np.array([5 * i for i in range(years + 1)])
ax.set_ylim([0, 18])
plt.xticks(x, x + year_shift)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.show()

"""for n in range(1, 5):
    # Basque country (ols)
    region = "Basque Country (Pais Vasco)"
    ols_ = Synth(region, year)
    ols_.fit(data, num_sv=n, method="Bayes")
    ols = ols_.estimate

    year = 15
    year_shift = 1955
    xlabel = "year"
    ylabel = "real per-capita GDP (1986 USD, thoudsand)"
    title = "Basque country study: # singular values = " + str(n)

    fig, ax = plt.subplots()
    ax.plot(ols_.raw, label='Basque Country', linewidth=1.75, color='k')
    ax.plot(ols_.estimate, '--', label='Robust synthetic Basque (Bayes)', linewidth=1.75, color='b')
    #ax.plot(abadie_y, '--', label='Original synthetic Basque', linewidth=1.75, color='g')

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
"""
