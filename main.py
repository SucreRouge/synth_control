import numpy as np
import pandas as pd
import robust_synth


def case_study(data, region, year, num_sv=2, method="Linear"):
    case = robust_synth.Synth(region, year=year, method=method)
    case.fit(data, num_sv=num_sv)
    case.vis_data()
    case.vis()


# BASQUE COUNTRY STUDY
df = pd.read_csv('basque.csv')
basque = df.pivot_table(values='gdpcap', index='regionname', columns='year')
basque = basque.drop('Spain (Espana)')
year = 15
region = "Basque Country (Pais Vasco)"
year_shift = 1955
xlabel = "year"
ylabel = "real per-capita GDP (1986 USD, thoudsand)"
title = "Basque country study"
case_study(basque, region, year, num_sv=2, method="Bayes")

# CALIFORNIA PROP 99 STUDY
df = pd.read_csv('cali.csv')
df = df[df['SubMeasureDesc'] == 'Cigarette Consumption (Pack Sales Per Capita)']
cali = df.pivot_table(values='Data_Value', index='LocationDesc', columns='Year')
#drop_list = [1, 2, 8, 9, 11, 20, 21, 22, 30, 32, 37, 47]
#data = data.drop(data.index[drop_list])
region = "California"
year = 18
year_shift = 1970
xlabel = "year"
ylabel = "per-capita cigarette sales (in packs)"
title = "Tobacco case study"
case_study(cali, region, year, num_sv=2, method="Bayes")
