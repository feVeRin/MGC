import numpy as np
from scipy import stats
import pandas as pd
from scipy.stats import ttest_rel
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
import Orange 
import matplotlib.pyplot as plt

#####friedman test

# Proposed = [58.28572, 85.49043, 36.04027, 74.875, 57.325, 37.27273, 67.12917, 82, 58.75663, 80.59361, 43.24561, 72.56823]
# Proposed_early = [54.57144, 82.94257, 35.97316, 72.875, 55.63125, 32.80993, 62.10526, 81.7, 55.37038, 79.38358, 40.61405, 70.48152]
# Comp1 = [45.7143, 61.81818, 27.71813, 62.875, 47.025, 28.42976, 61.91389, 77.5, 53.49211, 70.95889, 37.80703, 63.62179]
# Comp2 = [42.42857, 44.67704, 28.9933, 67.5, 42.19375, 25.20661, 63.44499, 63.35, 51.0053, 67.87672, 39.29826, 60.72232]
# Comp3 = [48.07142, 62.88278, 32.95301, 66, 46.9625, 27.19007, 39.97874444, 75.1, 52.53968, 71.32419, 38.42106, 63.4992]
# Comp4 = [47.71429, 54.37799, 31.34227, 65.875, 40.875, 24.29751, 35.35885, 72.5, 48.14813, 67.67124, 39.34211, 61.62119]


# result = stats.friedmanchisquare(Proposed, Comp1, Comp2, Comp3, Comp4)

# print(result)

# names = ["Proposed", "[1]", "[2]", "[3]", "[4]"]
# avranks = [1.00, 3.17,3.92, 2.83, 4.08]
# N = 12 # tested number of datasets
# cd = Orange.evaluation.compute_CD(avranks, N,alpha="0.05", test="bonferroni-dunn") #tested on 12 datasets 
# print("Critical Difference = ",cd)
# Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=5, textspace=1.5, cdmethod=0)
# plt.show();


#####Bonferroni-Dunn test

Dataset = 'Seyerlehner-Unique'

df = pd.read_csv('./'+ Dataset +'.csv', index_col=0)
res = []

if df['Proposed'].mean() > df['PE'].mean() and stats.ttest_rel(df['Proposed'], df['PE'])[1]<0.01:
    res.append('win*')
elif stats.ttest_rel(df['Proposed'], df['PE'])[1]>=0.01 and stats.ttest_rel(df['Proposed'], df['PE'])[1]<=0.05:
    res.append('win')
elif stats.ttest_rel(df['Proposed'], df['PE'])[1]>0.05:
    res.append('tie')
elif df['Proposed'].mean() < df['PE'].mean() and stats.ttest_rel(df['Proposed'], df['PE'])[1]<=0.05:
    res.append('lose')

algo_list = ['Proposed', 'c1', 'c2', 'c3', 'c4']
for i in range(len(algo_list)-1):
    for j in range(i+1, len(algo_list)):
        if df[algo_list[i]].mean() > df[algo_list[j]].mean() and stats.ttest_rel(df[algo_list[i]], df[algo_list[j]])[1]<0.01:
            res.append('win*')
        elif df[algo_list[i]].mean() > df[algo_list[j]].mean() and stats.ttest_rel(df[algo_list[i]], df[algo_list[j]])[1]>=0.01 and stats.ttest_rel(df[algo_list[i]], df[algo_list[j]])[1]<=0.05:
            res.append('win')
        elif stats.ttest_rel(df[algo_list[i]], df[algo_list[j]])[1]>0.05:
            res.append('tie')
        elif df[algo_list[i]].mean() < df[algo_list[j]].mean() and stats.ttest_rel(df[algo_list[i]], df[algo_list[j]])[1]<=0.05:
            res.append('lose')

print(res[0])
print(res[1:5])
print(res[5:8])
print(res[8:10])
print(res[10:11])





# import scipy.stats
# print(scipy.stats.f.ppf(q=1-0.05, dfn=4, dfd=44))
# crit = _
# scipy.stats.f.cdf(crit, dfn=3, dfd=39)