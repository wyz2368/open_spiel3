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

# plt.figure()
# # plt.title("NashConv Curves ", fontsize = 22)
#
# alpha_10_mean = genfromtxt('./data/alpha_10_mean.csv', delimiter=',')
# alpha_10_std = genfromtxt('./data/alpha_10_std.csv', delimiter=',')
#
# alpha_20_mean = genfromtxt('./data/alpha_20_mean.csv', delimiter=',')
# alpha_20_std = genfromtxt('./data/alpha_20_std.csv', delimiter=',')
#
# alpha_30_mean = genfromtxt('./data/alpha_30_mean.csv', delimiter=',')
# alpha_30_std = genfromtxt('./data/alpha_30_std.csv', delimiter=',')
#
# alpha_40_mean = genfromtxt('./data/alpha_40_mean.csv', delimiter=',')
# alpha_40_std = genfromtxt('./data/alpha_40_std.csv', delimiter=',')
#
# alpha_40_pro_mean = genfromtxt('./data/alpha_40_pro_mean.csv', delimiter=',')
# alpha_40_pro_std = genfromtxt('./data/alpha_40_pro_std.csv', delimiter=',')
#
# DO_mean = genfromtxt('./data/DO_mean.csv', delimiter=',')
# DO_std = genfromtxt('./data/DO_std.csv', delimiter=',')
#
#
#
# X = np.arange(1, 151)
#
# plt.plot(X, alpha_10_mean, color="C0", label='Size=10')
# plt.fill_between(X, alpha_10_mean+alpha_10_std, alpha_10_mean-alpha_10_std, alpha=0.1, color="C0")
#
# plt.plot(X, alpha_20_mean, color="C1", label='Size=20')
# plt.fill_between(X, alpha_20_mean+alpha_20_std, alpha_20_mean-alpha_20_std, alpha=0.1, color="C1")
#
# plt.plot(X, alpha_30_mean, color="C2", label='Size=30')
# plt.fill_between(X, alpha_30_mean+alpha_30_std, alpha_30_mean-alpha_30_std, alpha=0.1, color="C2")
#
# plt.plot(X, alpha_40_mean, color="C3", label='Size=40')
# plt.fill_between(X, alpha_40_mean+alpha_40_std, alpha_40_mean-alpha_40_std, alpha=0.1, color="C3")
#
# plt.plot(X, alpha_40_pro_mean, color="C4", label='Size=40 (pro)')
# plt.fill_between(X, alpha_40_pro_mean+alpha_40_pro_std, alpha_40_pro_mean-alpha_40_pro_std, alpha=0.1, color="C4")
#
# plt.plot(X, DO_mean, color="b", label='DO')
# plt.fill_between(X, DO_mean+DO_std, DO_mean-DO_std, alpha=0.1, color="b")
#
#
# plt.xticks(size = 17)
# plt.yticks(size = 17)
#
# plt.xlabel('Training Iterations', fontsize = 22)
# plt.ylabel('NashConv', fontsize = 19)
#
# plt.legend(loc="best")
#
# plt.show()



############### Plot etrace filter ###################


plt.figure()
# plt.title("NashConv Curves ", fontsize = 22)

etrace_30_mean = genfromtxt('./data/etrace_30_0.5_mean.csv', delimiter=',')
etrace_30_std = genfromtxt('./data/etrace_30_0.5_std.csv', delimiter=',')

etrace_40_mean = genfromtxt('./data/etrace_40_mean.csv', delimiter=',')
etrace_40_std = genfromtxt('./data/etrace_40_std.csv', delimiter=',')

DO_mean = genfromtxt('./data/DO_mean.csv', delimiter=',')
DO_std = genfromtxt('./data/DO_std.csv', delimiter=',')

axes = plt.gca()
axes.set_ylim([0,3])

X = np.arange(1, 151)

plt.plot(X, etrace_30_mean, color="C1", label='Size=30')
plt.fill_between(X, etrace_30_mean+etrace_30_std, etrace_30_mean-etrace_30_std, alpha=0.1, color="C1")

plt.plot(X, etrace_40_mean, color="C4", label='Size=40')
plt.fill_between(X, etrace_40_mean+etrace_40_std, etrace_40_mean-etrace_40_std, alpha=0.1, color="C4")

plt.plot(X, DO_mean, color="C0", label='DO')
plt.fill_between(X, DO_mean+DO_std, DO_mean-DO_std, alpha=0.1, color="C0")


plt.xticks(size = 17)
plt.yticks(size = 17)

plt.xlabel('Training Iterations', fontsize = 22)
plt.ylabel('NashConv', fontsize = 19)

plt.legend(loc="best")

plt.show()