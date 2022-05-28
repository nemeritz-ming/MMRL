import sys
from collections import defaultdict

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces

from histogram import combine, plot_bar

MAX_ACCOUNT_BALANCE = 2147483647
INITIAL_ACCOUNT_BALANCE = 1000000
Trading_Records = []

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.len = len(df)
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.INITIAL_ACCOUNT_BALANCE = INITIAL_ACCOUNT_BALANCE
        self.action_space = spaces.Box(
            low=np.array([0, 1]), high=np.array([3, 2]), dtype=np.float16)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8,), dtype=np.float16)

    def _next_observation(self):
        """
            load data
        """
        obs = np.array([
            self.df.loc[self.current_step, 'price'],
            self.df.loc[self.current_step, 'qty'],
            self.df.loc[self.current_step, 'quoteQty'],
            self.df.loc[self.current_step, 'time'],
            self.df.loc[self.current_step, 'isBuyerMaker']
        ], dtype=float)
        return obs

    def _take_action(self, actions):
        mid_price = (self.b1 + self.a1) / 2

        # check your previous limit orders
        if len(self.lob_record.keys()) != 0:
            for set_price in self.lob_record.keys():
                if self.lob_record[set_price] > 0 and set_price >= self.a1:
                    # execute buy one!
                    self.balance -= self.a1 * self.lob_record[set_price]
                    # self.balance += self.a1 * abs(self.lob_record[set_price]) * 0.0001

                    self.shares_held += self.lob_record[set_price]

                    # Trading_Records.append(f"{self.prev_time},{self.a1}, buy , {abs(self.lob_record[set_price])}, {self.shares_held}, {self.balance}")
                    # print(f'{self.a1}, buy , {self.lob_record[set_price]} , {self.shares_held}, {mid_price}, {self.balance}')

                    self.lob_record[set_price] = 0


                elif self.lob_record[set_price] < 0 and set_price <= self.b1:
                    # execute sell one!
                    self.balance += self.b1 * -self.lob_record[set_price]
                    # self.balance += self.b1 * abs(self.lob_record[set_price]) * 0.0001
                    self.shares_held -= -self.lob_record[set_price]

                    # Trading_Records.append(f"{self.prev_time},{self.b1}, sell , {abs(self.lob_record[set_price])}, {self.shares_held}, {self.balance}")
                    # print(f'{self.b1}, sell , {self.lob_record[set_price]} , {self.shares_held}, {mid_price}, {self.balance}')
                    self.lob_record[set_price] = 0


        # Set the current price
        mid_price = (self.b1 + self.a1) / 2

        for action in actions:
            action_type = action[0]
            amount = action[1]

            # Put one limit bid order at price a (buy stock)
            if action_type == 0:
                # check it is executed?
                if amount > self.a1:
                    self.balance -= self.a1
                    # self.balance += self.a1 * 0.0001
                    self.shares_held += 1
                    # Trading_Records.append(f"{self.prev_time},{self.a1}, buy , 1 , {self.shares_held}, {self.balance}")
                    # print(f'{self.a1}, buy , 1 , {self.shares_held}, {mid_price}, {self.balance}')

                else:
                    self.lob_record[amount] += 1

            # Put one limit ask order at price a (buy stock)
            elif action_type == 1:
                # check it is executed?
                if amount < self.b1:
                    # transaction executed
                    self.balance += self.b1
                    # self.balance += self.a1 * 0.0001

                    self.shares_held -= 1
                    # Trading_Records.append(f"{self.prev_time},{self.b1}, sell , 1 , {self.shares_held}, {self.balance}")
                    # print(f'{self.b1}, sell , -1 , {self.shares_held}, {mid_price}, {self.balance}')

                else:
                    self.lob_record[amount] -= 1

            elif action_type == 2:
                # clear your inventory
                self.balance += self.shares_held * mid_price
                tx = "sell" if self.shares_held > 0 else "buy"
                print(f'{mid_price}, {tx} , {self.shares_held} , 0, {mid_price}')
                # self.balance -= abs(self.shares_held) * mid_price * 0.0001
                Trading_Records.append(f"{self.prev_time},{mid_price}, {tx} ,{abs(self.shares_held)}, 0, {self.balance}")

                # transaction fee
                self.shares_held = 0


            self.net_worth = self.balance + self.shares_held * mid_price

            if self.net_worth > self.max_net_worth:
                self.max_net_worth = self.net_worth

            if self.shares_held == 0:
                self.cost_basis = 0

    def step(self, current_action):
        # Execute one time step within the environment
        previous_net_worth = self.net_worth
        done = False

        if self.current_step >= self.df.shape[0]:
            done = True
            return 0, 0, done, {}
        self.prev_time = self.df.loc[self.current_step, "time"]
        self.trade_record[self.prev_time] = defaultdict()
        self.trade_record[self.prev_time]["buy"] = set()
        self.trade_record[self.prev_time]["sell"] = set()
        if self.df.loc[self.current_step, 'isBuyerMaker']:
            self.trade_record[self.prev_time]["buy"].add(
                (self.df.loc[self.current_step, "price"], self.df.loc[self.current_step, "qty"]))
        else:
            self.trade_record[self.prev_time]["sell"].add(
                (self.df.loc[self.current_step, "price"], self.df.loc[self.current_step, "qty"]))

        self.current_step += 1
        if self.current_step < self.len - 1:
            while self.df.loc[self.current_step, "time"] == self.prev_time:
                if self.df.loc[self.current_step, "isBuyerMaker"]:
                    self.trade_record[self.prev_time]["buy"].add(
                        (self.df.loc[self.current_step, "price"], self.df.loc[self.current_step, "qty"]))
                else:
                    self.trade_record[self.prev_time]["sell"].add(
                        (self.df.loc[self.current_step, "price"], self.df.loc[self.current_step, "qty"]))
                self.current_step += 1
                if self.current_step > self.len - 1:
                    done = True
                    break
        else:
            done = True
        if len(self.trade_record[self.prev_time]["sell"]) != 0:
            self.a1 = max([a[0] for a in self.trade_record[self.prev_time]["sell"]])
        if len(self.trade_record[self.prev_time]["buy"]) != 0:
            self.b1 = min([a[0] for a in self.trade_record[self.prev_time]["buy"]])

        self._take_action(current_action)
        if self.net_worth <= 0:
            done = True

        if done:
            # print(info['inv'])
            self._take_action([[2, 0]])
        profit = self.net_worth - self.INITIAL_ACCOUNT_BALANCE
        reward = self.net_worth - previous_net_worth
        obs = self.trade_record[self.prev_time]

        return obs, reward, done, {"profit": profit, "inv": self.shares_held, "net_worth": self.net_worth,
                                   "time": self.prev_time, 'balance': self.balance,
                                   'a1': self.a1,
                                   'b1': self.b1, 'lob':self.lob_record}

    def reset(self, new_df=None):

        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.trade_record = defaultdict()
        self.current_step = 0
        self.prev_time = -1
        self.a1 = -1
        self.b1 = sys.maxsize
        self.lob_record = defaultdict(int)
        # pass test dataset to environment

        return self.step([[3, 0]])

if __name__ == '__main__':
    for day in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
                "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                "21", "22", "23", "24", "25", "26", "27", "28","29", "30"]:
        url = '/Users/ming/Documents/Project/data/trades/BTCUSDT-trades-2021-11-{0}.csv'.format(day)
        print('start at:', day, 'Nov.')
        dec_train1 = pd.read_csv(url)
        # print(dec_train1.columns)
        dec_train1.columns = ['trade_id', 'price', 'qty', 'quoteQty', 'time', 'isBuyerMaker']
        dec_train1 = dec_train1.drop(columns='trade_id')
        max_price = dec_train1['price'].max()
        min_price = dec_train1['price'].min()
        data_len = dec_train1.shape[0]
        seq_len = 2000
        num = int(data_len / seq_len)
        profit = []
        market_price = []
        my_data_sell = defaultdict(int)
        my_data_buy = defaultdict(int)
        my_profit = 0
        mid_price = 0
        n0 = 4
        for i in range(num):
            df1 = dec_train1.iloc[i * seq_len:(i + 1) * seq_len, :]
            print('from', i * seq_len, ' to ', (i + 1) * seq_len, ':')
            df1 = df1.reset_index(drop=False)
            env = StockTradingEnv(df1)
            action_bef = 0
            observation = env.reset(df1)
            my_data_sell = defaultdict(int)
            my_data_buy = defaultdict(int)
            done = False
            inv = []
            score = 0
            _, _, _, info = observation
            time = info['time']
            time_e = time + 10000
            s = 0
            while not done:
                trade, reward, done, info = observation
                time = info['time']
                my_data_buy, my_data_sell = combine(my_data_buy, my_data_sell, trade)
                while len(my_data_buy) == 0 or len(my_data_sell) == 0:
                    action = [[3, 0]]
                    observation = env.step(action)
                    trade, reward, done, info = observation
                    if trade == 0:
                        break
                    my_data_buy, my_data_sell = combine(my_data_buy, my_data_sell, trade)
                if done:
                    break
                a1 = info['a1']
                b1 = info['b1']
                my_inv = info['inv']
                market_price.append((a1 + b1) / 2)
                mid_price = market_price[-1]
                action_space = [(3, 0)]

                if time > time_e:
                    num_interval = 40
                    range_ = 0.06 * (max_price - min_price)
                    delta = range_ / num_interval
                    buy_ans, sell_ans = plot_bar(my_data_buy, my_data_sell, range_, num_interval, mid_price)
                    a = buy_ans[buy_ans['col2'] >= buy_ans[buy_ans['col2'] > 0]['col2'].quantile(0.9)]['price'].values
                    b = sell_ans[sell_ans['col2'] >= sell_ans[sell_ans['col2'] > 0]['col2'].quantile(0.9)][
                        'price'].values
                    target = np.intersect1d(a, b)
                    for sell_buy_price in target:
                        if abs(market_price[-1] - sell_buy_price) <= delta and abs(my_inv) <= 2:
                            action_space = [(0, sell_buy_price), (1, sell_buy_price)]
                            break
                        else:
                            action_space = [(3, 0)]

                observation = env.step(action_space)
                trade, reward, done, info = observation
                score += reward
                s += 1
            my_profit += score
            profit.append(my_profit)
            print(i, 'score for this episode: ', score)
        print('end at:', day)
#         with open("result_{0}.txt".format(day), "w") as output:
#             output.write(str(profit))


