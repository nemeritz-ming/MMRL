import sys
import warnings
from collections import defaultdict
import gym
import numpy as np
import pandas as pd
from gym import spaces
from histogram import combine, plot_bar
warnings.filterwarnings('ignore')


# some hyper-parameters
MAX_ACCOUNT_BALANCE = 2147483647
INITIAL_ACCOUNT_BALANCE = 1000000
Trading_Records = []
seq_len = 2000
range_coefficient = 0.06
num_interval = 5
quantile_val = 0.6
stop_coefficient = 0.95
market_order_fee = 0
limit_order_fee = 0
stop_loss_val = -500


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        # initialize parameters
        self.done = False
        self.df = df
        self.len = len(df)
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.INITIAL_ACCOUNT_BALANCE = INITIAL_ACCOUNT_BALANCE
        self.MAX_ACCOUNT_BALANCE = MAX_ACCOUNT_BALANCE
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.trade_record = defaultdict()
        self.current_step = 0
        self.prev_time = -1
        self.a1 = -1
        self.b1 = sys.maxsize
        self.lob_record = defaultdict(int)
        self.done = False
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

        # check your limit orders placed previously
        if len(self.lob_record.keys()) != 0:
            for set_price in self.lob_record.keys():
                if self.lob_record[set_price] > 0 and set_price > self.b1:
                    self.balance -= self.b1 * self.lob_record[set_price]
                    self.balance += self.a1 * abs(self.lob_record[set_price]) * limit_order_fee
                    self.shares_held += self.lob_record[set_price]
                    self.lob_record[set_price] = 0


                elif self.lob_record[set_price] < 0 and set_price < self.a1:
                    self.balance += self.a1 * -self.lob_record[set_price]
                    self.balance += self.b1 * abs(self.lob_record[set_price]) * limit_order_fee
                    self.shares_held -= -self.lob_record[set_price]
                    self.lob_record[set_price] = 0

                else:
                    continue

        # Set the current price
        mid_price = (self.b1 + self.a1) / 2

        # do actions
        for action in actions:
            action_type = action[0]
            amount = action[1]

            # Put one limit bid order at price a (buy stock)
            if action_type == 0:
                # check it is executed?
                if amount > self.b1:
                    self.balance -= self.b1
                    self.balance += self.a1 * limit_order_fee
                    self.shares_held += 1

                else:
                    self.lob_record[amount] += 1

            # Put one limit ask order at price a (buy stock)
            elif action_type == 1:
                # check it is executed?
                if amount < self.a1:
                    # transaction executed
                    self.balance += self.a1
                    self.balance += self.a1 * limit_order_fee

                    self.shares_held -= 1

                else:
                    self.lob_record[amount] -= 1

            elif action_type == 2:
                # clear your inventory
                self.balance += self.shares_held * mid_price
                self.balance -= abs(self.shares_held) * mid_price * market_order_fee

                self.shares_held = 0

            self.net_worth = self.balance + self.shares_held * mid_price

            if self.net_worth > self.max_net_worth:
                self.max_net_worth = self.net_worth


    def step(self, current_action):
        # Execute step with the same time within the environment
        previous_net_worth = self.net_worth
        self.done = False

        # collect data at the same time
        if self.current_step >= self.df.shape[0]:
            self.done = True
            return 0, 0, self.done, {}
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
                    self.done = True
                    break
        else:
            self.done = True
        if len(self.trade_record[self.prev_time]["sell"]) != 0:
            self.a1 = max([a[0] for a in self.trade_record[self.prev_time]["sell"]])
        if len(self.trade_record[self.prev_time]["buy"]) != 0:
            self.b1 = min([a[0] for a in self.trade_record[self.prev_time]["buy"]])

        # take actions
        self._take_action(current_action)

        # set stop criterion
        if self.net_worth <= 0 or self.balance < 0 or self.balance > self.MAX_ACCOUNT_BALANCE:
            self.done = True
        # at the end of trade we close the position
        if self.done:
            self._take_action([[2, 0]])

        # return the information
        profit = self.net_worth - self.INITIAL_ACCOUNT_BALANCE
        reward = self.net_worth - previous_net_worth
        obs = self.trade_record[self.prev_time]

        return obs, reward, self.done, {"profit": profit,
                                        "inv": self.shares_held,
                                        "net_worth": self.net_worth,
                                        "time": self.prev_time,
                                        'balance': self.balance,
                                        'a1': self.a1,
                                        'b1': self.b1,
                                        'lob': self.lob_record}

    def reset(self, new_df=None):

        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.trade_record = defaultdict()
        self.current_step = 0
        self.prev_time = -1
        self.a1 = -1
        self.b1 = sys.maxsize
        self.lob_record = defaultdict(int)
        self.done = False
        # pass test dataset to environment

        return self.step([])


#
if __name__ == '__main__':
    my_profit = 0
    profit = []
    for day in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
                "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"]:
        # read data
        url = '../data/trades/BTCUSDT-trades-2021-11-{0}.csv'.format(day)
        # print('start at:', day, 'Nov.')
        dec_train1 = pd.read_csv(url)
        # print(dec_train1.columns)
        dec_train1.columns = ['trade_id', 'price', 'qty', 'quoteQty', 'time', 'isBuyerMaker']
        dec_train1 = dec_train1.drop(columns='trade_id')
        max_price = dec_train1['price'].max()
        min_price = dec_train1['price'].min()
        data_len = dec_train1.shape[0]
        num = int(data_len / seq_len)

        # record profit and market price
        market_price = []

        for i in range(num):
            # pick the trading episode and create the trading environment
            df1 = dec_train1.iloc[i * seq_len:(i + 1) * seq_len, :]
            df1 = df1.reset_index(drop=False)
            env = StockTradingEnv(df1)
            observation = env.reset(df1)

            # create trade volume histogram
            my_data_sell = defaultdict(int)
            my_data_buy = defaultdict(int)
            done = False
            inv = []
            score = 0
            _, _, _, info = observation

            # record the profit
            p = -1

            while not done:
                # read the observations, profit
                trade, _, done, info = observation
                p = info['profit']

                # record the number of non-executed limit orders
                not_trade = sum([abs(k) for k in info['lob'].values()])

                # update the histogram
                my_data_buy, my_data_sell = combine(my_data_buy, my_data_sell, trade)
                while len(my_data_buy) == 0 or len(my_data_sell) == 0:
                    action = []
                    observation = env.step(action)
                    trade, reward, done, info = observation
                    if trade == 0:
                        break
                    my_data_buy, my_data_sell = combine(my_data_buy, my_data_sell, trade)
                if done:
                    break

                # record the trading information
                a1 = info['a1']
                b1 = info['b1']
                my_inv = info['inv']
                market_price.append((a1 + b1) / 2)
                mid_price = market_price[-1]
                range_ = range_coefficient * (max_price - min_price)
                delta = range_ / num_interval
                action_space = []

                # create channel
                buy_ans, sell_ans = plot_bar(my_data_buy, my_data_sell, range_, num_interval, mid_price)
                a = buy_ans[buy_ans['col2'] >= buy_ans[buy_ans['col2'] > 0]['col2'].quantile(quantile_val)][
                    'price'].values
                b = sell_ans[sell_ans['col2'] >= sell_ans[sell_ans['col2'] > 0]['col2'].quantile(quantile_val)][
                    'price'].values

                # select target price
                target = np.intersect1d(a, b)
                for sell_buy_price in target:
                    if abs(market_price[-1] - sell_buy_price) <= delta:
                        # if current price is higher, we put one limit sell order
                        if market_price[-1] > sell_buy_price:
                            action_space = [(1, a1)]
                        # if current price is lower, we put one limit buy order
                        else:
                            action_space = [(0, b1)]
                        break
                observation = env.step(action_space)
                trade, reward_, done, info = observation
                score += reward_

                # stopping strategy, we did not consider the transaction fee here
                if info['profit'] > 0:
                    if info['profit'] > p or info['profit'] > p * stop_coefficient:
                        p = info['profit']
                    else:
                        done = True
                if score < stop_loss_val:
                    done = True
            my_profit += score
            profit.append(my_profit)

    # with open("CB_result_without_fee.txt".format(day), "w") as output:
    #     output.write(str(profit))