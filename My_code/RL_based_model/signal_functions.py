import collections as c
import warnings
from statistics import mean
from scipy.stats import norm
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def EMA(mid_price):
    N = len(mid_price)
    start = mid_price[0]
    for i in range(1, N):
        start = (2 * mid_price[i] + (N - 1) * start) / (N + 1)
    return start


def MACD(mid_price):
    return EMA(mid_price[-26:]) - EMA(mid_price[-9:])


def Aroon(mid_price):
    return (np.argmax(mid_price) + 1) / len(mid_price)


def AO(mid_price):
    return sum(mid_price[-34:]) / 34 - sum(mid_price[-5:]) / 5


def volatility(data):
    diff = []
    for i in range(1, len(data)):
        diff.append(data[i] / data[i - 1] - 1)
    return np.std(diff)


def minimum_increase_spread(data):
    res = 1
    for i in range(1, len(data)):
        if data[i] / data[i - 1] > 1:
            res = min(data[i] / data[i - 1] - 1, res)


def minimum_decrease_spread(data):
    res = -1
    for i in range(1, len(data)):
        if data[i] / data[i - 1] < 1:
            res = max(data[i] / data[i - 1] - 1, res)


def transition_times(data):
    res = 0
    for i in range(1, len(data) - 1):
        if data[i] / data[i - 1] < 1 and data[i + 1] / data[i] > 1:
            res += 1
        elif data[i] / data[i - 1] > 1 and data[i + 1] / data[i] < 1:
            res += 1
    return res / (len(data) - 2)


def rsi_increase(data):
    increase = []
    decrease = []
    for i in range(1, len(data)):
        if data[i] / data[i - 1] > 1:
            increase.append(data[i] - data[i - 1])
        elif data[i] / data[i - 1] < 1:
            decrease.append(data[i] - data[i - 1])
    if len(increase) != 0 and len(decrease) != 0:
        return np.sum(np.array(increase)) / (np.sum(np.array(increase)) - np.sum(np.array(decrease)))
    elif len(decrease) == 0 and len(increase) != 0:
        return 1
    elif len(increase) == 0 and len(decrease) != 0:
        return 0
    else:
        return -1


def rsi_decrease(data):
    increase = []
    decrease = []
    for i in range(1, len(data)):
        if data[i] / data[i - 1] > 1:
            increase.append(data[i] - data[i - 1])
        elif data[i] / data[i - 1] < 1:
            decrease.append(data[i] - data[i - 1])
    if len(increase) != 0 and len(decrease) != 0:
        return -np.sum(np.array(decrease)) / (np.sum(np.array(increase)) - np.sum(np.array(decrease)))
    elif len(decrease) == 0 and len(increase) != 0:
        return 0
    elif len(increase) == 0 and len(decrease) != 0:
        return 1
    else:
        return -1


def history_buy_price_profit_rate_distribution(A1, B1):
    result_profit = c.defaultdict(list)
    result_win = c.defaultdict(int)
    for i in range(len(A1)):
        for j in range(i, len(B1)):
            if B1[j] > A1[i]:
                result_profit[A1[i]].append(j - i)
                result_win[A1[i]] = result_win[A1[i]] + 1
                break
    for keys in result_profit:
        result_profit[keys] = mean(result_profit[keys])
    return result_win, result_profit


def history_sell_price_profit_rate_distribution(A1, B1):
    result_profit = c.defaultdict(list)
    result_win = c.defaultdict(int)
    for i in range(len(B1)):
        for j in range(i, len(A1)):
            if B1[i] > A1[j]:
                result_profit[B1[i]].append(j - i)
                result_win[B1[i]] = result_win[B1[i]] + 1
                break
    for keys in result_profit:
        result_profit[keys] = mean(result_profit[keys])
    return result_win, result_profit


def top(l, percent):
    idx = int(len(l) * percent)
    return sorted(l)[-idx:]


def data_generator(length):
    ans = []
    for i in range(length):
        ans.append(round(np.random.random() * 8 - 4 + 100))
    return np.array(ans)


def data_generator_brown(length, x=100, delta=0.2, dt=0.1):
    res = []
    for k in range(length):
        x = x + norm.rvs(scale=delta ** 2 * dt)
        res.append(x)
    return np.array(res)


def lr_pred(series):
    x = np.array(list(range(len(series))))
    x = np.array([[i + 1] for i in x])
    model = LinearRegression().fit(x, series)
    return model.coef_[0]


def ar_model_pred(series):
    model = AutoReg(series, lags=1)
    model_fit = model.fit()
    return model_fit.predict(len(series), len(series))


def arima_pred(series, n):
    model = ARIMA(series, order=(2, 0, 1))
    model_fit = model.fit()
    return model_fit.forecast(n)


def win_prob_ask(mid_price, A1, delta_p):
    count = 0
    for i in range(len(mid_price) - 1):
        depth = mid_price[i] + delta_p
        if A1[i + 1] <= depth:
            count += 1
    return count / (len(mid_price) - 1)


def win_prob_bid(mid_price, B1, delta_p):
    count = 0
    for i in range(len(mid_price) - 1):
        depth = mid_price[i] + delta_p
        if B1[i + 1] >= depth:
            count += 1
    return count / (len(mid_price) - 1)
