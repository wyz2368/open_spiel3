from numpy import genfromtxt
import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

from scipy.signal import savgol_filter
# yhat = savgol_filter(y, 51, 2) # window size 51, polynomial order 3

window_size = 55
order = 2

########### Plot alpha rank filter ############

plt.figure()
# plt.title("NashConv Curves ", fontsize = 22)


Mike_fic_mean = genfromtxt('./data/2Nash_merged_csv/dqn_fic_Mike_mean.csv', delimiter=',')
Mike_fic_std = genfromtxt('./data/2Nash_merged_csv/dqn_fic_Mike_std.csv', delimiter=',')

dqn_do_mean = genfromtxt('./data/2Nash_merged_csv/dqn_DO_mean.csv', delimiter=',')
dqn_do_std = genfromtxt('./data/2Nash_merged_csv/dqn_DO_std.csv', delimiter=',')

Mike_prd_mean = genfromtxt('./data/2Nash_merged_csv/dqn_prd_Mike_mean.csv', delimiter=',')
Mike_prd_std = genfromtxt('./data/2Nash_merged_csv/dqn_prd_Mike_std.csv', delimiter=',')

axes = plt.gca()
axes.set_ylim([0.5,3])

X = np.arange(1, 151)

plt.plot(X, Mike_fic_mean, color="C2", label='Uniform')
plt.fill_between(X, Mike_fic_mean+Mike_fic_std, Mike_fic_mean-Mike_fic_std, alpha=0.1, color="C2")

plt.plot(X, dqn_do_mean, color="C1", label='DO')
plt.fill_between(X, dqn_do_mean+dqn_do_std, dqn_do_mean-dqn_do_std, alpha=0.1, color="C1")

plt.plot(X, Mike_prd_mean, color="C0", label='PRD')
plt.fill_between(X, Mike_prd_mean+Mike_prd_std, Mike_prd_mean-Mike_prd_std, alpha=0.1, color="C0")


plt.xticks(size = 17)
plt.yticks(size = 17)

plt.xlabel('Number of Iterations', fontsize = 22)
plt.ylabel('NashConv', fontsize = 19)

plt.legend(loc="best")

plt.show()

