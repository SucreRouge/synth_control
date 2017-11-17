from __future__ import division
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import robust_synth
import synth_functions


def synth_plots_bayes(obs, bayes, sigma_hat, abadie, title, xlabel, ylabel, region, year, year_shift, loc, up_lim):
    fig, ax = plt.subplots()
    ax.plot(obs, label=region, linewidth=1.75, color='k')
    ax.plot(bayes, '--', label='Synthetic ' + region + " (Bayes)", linewidth=1.75, color='b')
    ax.plot(abadie, '--', label='Synthetic ' + region +
            " (Abadie et. al)", linewidth=1.75, color='g')
    x_ = np.linspace(0, len(bayes) - 1, len(bayes))
    clr1 = 'lightcyan'
    clr2 = 'paleturquoise'
    upper = bayes + sigma_hat
    lower = bayes - sigma_hat
    ax.fill_between(x_, bayes, upper, facecolor=clr1, edgecolor=clr2, interpolate=True)
    ax.fill_between(x_, bayes, lower, facecolor=clr1, edgecolor=clr2, interpolate=True)
    legend = ax.legend(loc=loc, shadow=True, prop={'size': 9.5})
    frame = legend.get_frame()
    frame.set_facecolor('0.925')
    ax.plot([year, year], [0, up_lim], '--', linewidth=1.5, color='r')
    years = int(np.floor(obs.shape[0] / 5))
    x = np.array([5 * i for i in range(years + 1)])
    ax.set_ylim([0, up_lim])
    plt.xticks(x, x + year_shift)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def synth_plots(obs, linear, ridge, lasso, abadie, title, xlabel, ylabel, region, year, year_shift, loc, upper):
    fig, ax = plt.subplots()
    ax.plot(obs, label=region, linewidth=1.75, color='k')
    ax.plot(linear, '--', label='Synthetic ' + region + " (linear)", linewidth=1.75, color='b')
    ax.plot(ridge, '--', label='Synthetic ' + region + " (ridge)", linewidth=1.75, color='g')
    ax.plot(lasso, '--', label='Synthetic ' + region +
            " (lasso)", linewidth=1.75, color='darkorange')
    ax.plot(abadie, '--', label='Synthetic ' + region +
            " (Abadie et. al)", linewidth=1.75, color='gray')
    legend = ax.legend(loc=loc, shadow=True, prop={'size': 9.5})
    frame = legend.get_frame()
    frame.set_facecolor('0.925')
    ax.plot([year, year], [0, upper], '--', linewidth=1.5, color='r')
    years = int(np.floor(obs.shape[0] / 5))
    x = np.array([5 * i for i in range(years + 1)])
    ax.set_ylim([0, upper])
    plt.xticks(x, x + year_shift)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


# BASQUE COUNTRY STUDY
df = pd.read_csv('basque.csv')
basque = df.pivot_table(values='gdpcap', index='regionname', columns='year')
basque = basque.drop('Spain (Espana)')
year = 15
region = "Basque Country (Pais Vasco)"
year_shift = 1955
xlabel = "year"
ylabel = "real per-capita GDP (1986 USD, thoudsand)"
title = "Basque Country study"

# method = regression method used {Linear, Ridge, Lasso, Bayes}
# num_sv = number of singular values to include (i.e. rank(hat(M)) = num_sv)
# year = year of intervention
# region = name of treatment unit
linear = robust_synth.Synth(region, year=year, num_sv=2, method="Linear")
ridge = robust_synth.Synth(region, year=year, num_sv=2, method="Ridge")
lasso = robust_synth.Synth(region, year=year, num_sv=2, method="Lasso")

linear.fit(basque)
ridge.fit(basque)
lasso.fit(basque)
obs = linear.orig  # the actual observed trajectory

# predicted means (pre + post intervention) using linear/ridge/lasso regression
linear_mean = linear.mean
ridge_mean = ridge.mean
lasso_mean = lasso.mean
abadie = np.array([3.70275826,   3.85377741,   3.99618902,   4.02922158,
                   4.05951172,   4.3787652,   4.73286861,   4.98746136,
                   5.22186476,   5.29834131,   5.3619788,   5.44839733,
                   5.52300483,   5.76061456,   5.99301375,   6.13783742,
                   6.29426509,   6.62068884,   6.93289289,   7.08693884,
                   7.22791833,   7.22057226,   7.21105396,   7.07458287,
                   7.05725888,   7.12924607,   7.23436873,   7.32529425,
                   7.42182591,   7.51631585,   7.61009202,   8.1179103,
                   8.62355603,   9.08677821,   9.54553342,   9.7882537,
                   10.03771928,   9.83822235,   9.63904542,   9.98791378,
                   10.30394025,  10.5384924,  10.99881349])


synth_plots(obs=obs, linear=linear_mean, ridge=ridge_mean, lasso=lasso_mean, abadie=abadie, title=title, xlabel=xlabel,
            ylabel=ylabel, region="Basque", year=year, year_shift=year_shift, loc="lower right", upper=12)

for n in range(4, 7):
    bayes = robust_synth.Synth(region, year=year, method="bayes", num_sv=n, prior_param=10)
    bayes.fit(basque)
    obs = bayes.orig
    bayes_mean = bayes.mean
    title = "Basque Country study: # singular values = " + str(n)
    synth_plots_bayes(obs=obs, bayes=bayes_mean, sigma_hat=bayes.sigma_hat, abadie=abadie, title=title, xlabel=xlabel,
                      ylabel=ylabel, region="Basque", year=year, year_shift=year_shift, loc="lower right", up_lim=14)


# CALIFORNIA PROP 99 STUDY
df = pd.read_csv('cali.csv')
df = df[df['SubMeasureDesc'] == 'Cigarette Consumption (Pack Sales Per Capita)']
cali = df.pivot_table(values='Data_Value', index='LocationDesc', columns='Year')
# drop_list = [1, 2, 8, 9, 11, 20, 21, 22, 30, 32, 37, 47]
# data = data.drop(data.index[drop_list])
region = "California"
year = 18
year_shift = 1970
xlabel = "year"
ylabel = "per-capita cigarette sales (in packs)"
title = "Tobacco case study"

linear = robust_synth.Synth(region, year=year, num_sv=3, method="Linear")
ridge = robust_synth.Synth(region, year=year, num_sv=3, method="Ridge")
lasso = robust_synth.Synth(region, year=year, num_sv=3, method="Lasso")
linear.fit(cali)
ridge.fit(cali)
lasso.fit(cali)
obs = linear.orig
linear_mean = linear.mean
ridge_mean = ridge.mean
lasso_mean = lasso.mean
abadie = np.array([116.8072,
                   118.6901,
                   124.1291,
                   125.2378,
                   126.8470,
                   127.0187,
                   127.7564,
                   125.6422,
                   124.9258,
                   122.7328,
                   120.2739,
                   119.9707,
                   116.6876,
                   111.0581,
                   103.0856,
                   102.9847,
                   99.5339,
                   99.4777,
                   91.3973,
                   89.8195,
                   87.3677,
                   81.9650,
                   81.4859,
                   81.0874,
                   80.6558,
                   78.4596,
                   77.4590,
                   77.7413,
                   74.3700,
                   73.5093,
                   67.2664])
synth_plots(obs=obs, linear=linear_mean, ridge=ridge_mean, lasso=lasso_mean, abadie=abadie, title=title, xlabel=xlabel,
            ylabel=ylabel, region="California", year=year, year_shift=year_shift, loc="lower left", upper=140)

"""for n in range(1, 4):
    bayes = robust_synth.Synth(region, year=year, method="bayes", num_sv=n, prior_param=15)
    bayes.fit(cali)
    obs = bayes.orig
    bayes_mean = bayes.mean
    title = "Tobacco case study: # singular values = " + str(n)
    synth_plots_bayes(obs=obs, bayes=bayes_mean, sigma_hat=bayes.sigma_hat, abadie=abadie, title=title, xlabel=xlabel,
                      ylabel=ylabel, region="California", year=year, year_shift=year_shift, loc="lower left", up_lim=140)
"""


# GERMANY REUNIFICATION
df = pd.read_csv("germany.csv")
germany = df.pivot_table(values='gdp', index='country', columns='year')
year = 30
region = "West Germany"
year_shift = 1960
xlabel = "year"
ylabel = "Per Capita GDP (PPP, 2002 USD)"
title = "German Reunification Study"

for n in range(1, 5):
    linear = robust_synth.Synth(region, year=year, num_sv=n, method="Linear")
    ridge = robust_synth.Synth(region, year=year, num_sv=n, method="Ridge")
    lasso = robust_synth.Synth(region, year=year, num_sv=n, method="Lasso")
    linear.fit(germany)
    ridge.fit(germany)
    lasso.fit(germany)
    obs = linear.orig
    linear_mean = linear.mean
    ridge_mean = ridge.mean
    lasso_mean = lasso.mean
    synth_plots(obs=obs, linear=linear_mean, ridge=ridge_mean, lasso=lasso_mean, abadie=[], title=title, xlabel=xlabel,
                ylabel=ylabel, region="West Germany", year=year, year_shift=year_shift, loc="lower right", upper=35000)
