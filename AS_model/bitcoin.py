import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
import random
import warnings

from fbm import FBM
import os

warnings.filterwarnings('ignore')


price_list = []
for t in range(1, 30):
    t = '0' + str(t) if t < 10 else t
    url = '../data/LOB/BTCUSDT_S_DEPTH_202111{0}.csv'.format(t)
    data = pd.read_csv(url)
    df = data[['ap1', 'bp1']]
    df['mid'] = (df.loc[:,'ap1'] + df.loc[:,'bp1'])/2
    selected_data = [df.loc[:, 'mid'].values[i] for i in range(1, len(df['mid'].values), int(len(df['mid'].values)/(2000)))]
    price_list = price_list +selected_data


# Parameters for mid price simulation:

T = 1.0  # time
sigma = 1  # volatility
# M = 100  # number of time steps
Sim = 1  # number of simulations
gamma = 0.1  # risk aversion
k = 1.5
A = 100000
I = 1

# Results:

AverageSpread = []
Profit = []
Std = []
s = price_list
M = len(s)
S = np.zeros((M + 1, I))
Bid = np.zeros((M + 1, I))
Ask = np.zeros((M + 1, I))
q = np.zeros((M + 1, I))
w = np.zeros((M + 1, I))
equity = [0] * (M + 1)

for i in range(1, Sim + 1):

    dt = T / M  # time step
    t = list(range(M))

    ReservPrice = np.zeros((M + 1, I))
    spread = np.zeros((M + 1, I))
    deltaB = np.zeros((M + 1, I))
    deltaA = np.zeros((M + 1, I))

    equity = [0]*(M + 1)
    reserve_relation = np.zeros((M + 1, I))

    S[0] = s[0]
    ReservPrice[0] = s[0]
    Bid[0] = s[0]
    Ask[0] = s[0]
    spread[0] = 0
    deltaB[0] = 0
    deltaA[0] = 0
    q[0] = 0  # position
    w[0] = 0  # wealth
    equity[0] = 0

    for t in range(1, M + 1):
        z = np.random.standard_normal(I)
        S[t] = s[t-1]
        ReservPrice[t] = S[t] - q[t - 1] * gamma * (sigma ** 2) * (T - t / float(M))
        spread[t] = gamma * (sigma ** 2) * (T - t / float(M)) + (2 / gamma) * math.log(1 + (gamma / k))
        Bid[t] = ReservPrice[t] - spread[t] / 2.
        Ask[t] = ReservPrice[t] + spread[t] / 2.

        deltaB[t] = S[t] - Bid[t]
        deltaA[t] = Ask[t] - S[t]

        lambdaA = A * np.exp(-k * deltaA[t])
        ProbA = lambdaA * dt
        fa = random.random()

        lambdaB = A * np.exp(-k * deltaB[t])
        ProbB = lambdaB * dt
        fb = random.random()

        if ProbB > fb and ProbA < fa:
            q[t] = q[t - 1] + 1
            w[t] = w[t - 1] - Bid[t]


        if ProbB < fb and ProbA > fa:
            q[t] = q[t - 1] - 1
            w[t] = w[t - 1] + Ask[t]


        if ProbB < fb and ProbA < fa:
            q[t] = q[t - 1]
            w[t] = w[t - 1]
        if ProbB > fb and ProbA > fa:
            q[t] = q[t - 1]
            w[t] = w[t - 1] - Bid[t]
            w[t] = w[t] + Ask[t]


        equity[t] = (w[t] + q[t] * S[t]).tolist()[0]
    AverageSpread.append(spread.mean())
    Profit.append(equity[-1])
    Std.append(equity[-1])
    # print(equity)
    equity = [equity[i] for i in
                     range(1, len(equity), 20)]
    print(equity)

print("                   Results              ")
print("----------------------------------------")

print("%14s %21s" % ('statistic', 'value'))
print(40 * "-")
print("%14s %20.5f" % ("Average spread :", np.array(AverageSpread).mean()))
print("%16s %20.5f" % ("Profit :", np.array(Profit).mean()))
print("%16s %20.5f" % ("Std(Profit) :", np.array(Std).std()))

# Plots:

x = np.linspace(0., T, num=(M + 1))
fig = plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)  # number of rows, number of  columns, number of the subplot
plt.plot(x, S[:], lw=1., label='S')
plt.plot(x, Ask[:], lw=1., label='Ask')
plt.plot(x, Bid[:], lw=1., label='Bid')
# plt.plot(x, ReservPrice[:], lw=1, label='ReserveP')
plt.grid(True)
plt.legend(loc=0)
plt.ylabel('P')
plt.title('Prices')
plt.subplot(2, 1, 2)
plt.plot(x, q[:], 'g', lw=1., label='q')  # plot 2 lines
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('Time')
plt.ylabel('Position')
plt.show()

# Histogram of profit:

# plt.figure(figsize=(7, 5))
# plt.hist(np.array(Profit), label=['hitogram'], bins=100)
# plt.legend(loc=0)
# plt.grid(True)
# plt.xlabel('pnl')
# plt.ylabel('number of values')
# plt.title('Histogram')
# plt.show()

# PNL:
# with open("result_AS_model.txt", "w") as output:
#     output.write(str(equity))

plt.figure(figsize=(7, 5))
plt.plot(np.array(list(range(len(equity))))/len(equity),np.array(equity), label='AS_model')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('pnl')
plt.ylabel('number of values')
plt.title('Profit')
plt.show()
