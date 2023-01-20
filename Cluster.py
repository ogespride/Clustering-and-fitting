

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.optimize as opt
from scipy.optimize import curve_fit
import seaborn as sns
import err_ranges as err

"""This is a function that reads csv file and returns an output of the dataframe"""
def Worldbank(filename,countries,columns,indicator):
    df = pd.read_csv(filename,skiprows=4)
    df = df[df['Indicator Name'] == indicator]
    df = df[columns]
    df.set_index('Country Name', inplace = True)
    #df = df.loc[countries]
    df = df.dropna()
    return df,df.transpose()

#Showing dataname, countries, climate change indicator and period of years being compared.
filename = 'API_19_DS2_en_csv_v2_4773766.csv'
countries = ['China','United States','Nigeria','Afghanistan']
columns = ['Country Name', '1990','2018']
indicators = ['CO2 emissions (metric tons per capita)']


df_year_co2,df_country_co2= Worldbank(filename,countries,columns,indicators[0])

#plot a cluster
clustering = df_year_co2.plot('1990', '2018', kind='scatter')

#Number of values
cluster = df_year_co2.values
print(cluster)

#Plot showing K-MEANS
plt.figure()
sse = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=10, random_state=0)
    kmeans.fit(cluster)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 10), sse)
plt.title('Elbow cluster')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

#K-means showing Number of iteration as Numpy arrays
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(cluster)
print(y_kmeans)

plt.scatter(cluster[y_kmeans == 0, 0], cluster[y_kmeans == 0, 1], s = 50, c = 'yellow',label = 'label 0')
plt.scatter(cluster[y_kmeans == 1, 0], cluster[y_kmeans == 1, 1], s = 50, c = 'red',label = 'label 1')
plt.scatter(cluster[y_kmeans == 2, 0], cluster[y_kmeans == 2, 1], s = 50, c = 'green',label = 'label 2')
#plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 50, c = 'blue',label = 'Iabel 3')
#plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 50, c = 'black',label = 'Iabel 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 10, c = 'black', label = 'Centroids')
plt.legend()
plt.show()

#MODEL FITING
#define the objective function
def linear(x, a, b):
        s = a + b*x
        return s
# create a few points with normal distributed random errors
xarr = np.linspace(0.0, 5.0, 15)
yarr = linear(xarr, 1.0, 0.2)
ymeasure = yarr + np.random.normal(0.0, 0.5, len(yarr))

lin_list = []
for x, ym in zip(xarr, ymeasure):
    lin_list.append([x, ym])

df_lin = pd.DataFrame(lin_list, columns=["1990", "2018"])
print(df_lin)

param, covar = opt.curve_fit(linear, df_year_co2["1990"],df_year_co2["2018"])
plt.figure()
plt.plot(df_year_co2["1990"], df_year_co2["2018"], "go", label="co2 data")
plt.plot(df_year_co2["1990"], linear(df_year_co2["1990"], *param), label="fit")
plt.plot(df_year_co2["2018"], linear(df_year_co2["2018"], 1.0, 0.2), label="a = 1.0+0.02*x")
plt.xlabel("1990")
plt.ylabel("2018")
plt.legend()
plt.show()







