import random
from collections import defaultdict
import gym
import numpy as np
import pandas as pd
from gym import spaces
import matplotlib.pyplot as plt
from RLstrats.simple_DQN import PolicyGradientAgent
from RLstrats.votality import volatility, transition_times, rsi_increase, rsi_decrease, \
    history_buy_price_profit_rate_distribution, history_sell_price_profit_rate_distribution,Aroon,AO
import torch
MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 2000
MAX_VOLUME = 400e8
MAX_AMOUNT = 3e10
MAX_STEPS = 20000
MAX_DAY_CHANGE = 1
INITIAL_ACCOUNT_BALANCE = 10


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

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8,), dtype=np.float16)

    def _next_observation(self):
        """
            data = data.loc[:, ['bid_price1',
       'ask_price1', 'bid_price2', 'ask_price2', 'bid_size1', 'ask_size1',
       'bid_size2', 'ask_size2']]
        """
        obs = np.array([
            self.df.loc[self.current_step, 'ask_price10'],
            self.df.loc[self.current_step, 'ask_price9'],
            self.df.loc[self.current_step, 'ask_price8'],
            self.df.loc[self.current_step, 'ask_price7'],
            self.df.loc[self.current_step, 'ask_price6'],
            self.df.loc[self.current_step, 'ask_price5'],
            self.df.loc[self.current_step, 'ask_price4'],
            self.df.loc[self.current_step, 'ask_price3'],
            self.df.loc[self.current_step, 'ask_price2'],
            self.df.loc[self.current_step, 'ask_price1'],
            self.df.loc[self.current_step, 'bid_price1'],
            self.df.loc[self.current_step, 'bid_price2'],
            self.df.loc[self.current_step, 'bid_price3'],
            self.df.loc[self.current_step, 'bid_price4'],
            self.df.loc[self.current_step, 'bid_price5'],
            self.df.loc[self.current_step, 'bid_price6'],
            self.df.loc[self.current_step, 'bid_price7'],
            self.df.loc[self.current_step, 'bid_price8'],
            self.df.loc[self.current_step, 'bid_price9'],
            self.df.loc[self.current_step, 'bid_price10'],
            self.df.loc[self.current_step, 'ask_size10'],
            self.df.loc[self.current_step, 'ask_size9'],
            self.df.loc[self.current_step, 'ask_size8'],
            self.df.loc[self.current_step, 'ask_size7'],
            self.df.loc[self.current_step, 'ask_size6'],
            self.df.loc[self.current_step, 'ask_size5'],
            self.df.loc[self.current_step, 'ask_size4'],
            self.df.loc[self.current_step, 'ask_size3'],
            self.df.loc[self.current_step, 'ask_size2'],
            self.df.loc[self.current_step, 'ask_size1'],
            self.df.loc[self.current_step, 'bid_size1'],
            self.df.loc[self.current_step, 'bid_size2'],
            self.df.loc[self.current_step, 'bid_size3'],
            self.df.loc[self.current_step, 'bid_size4'],
            self.df.loc[self.current_step, 'bid_size5'],
            self.df.loc[self.current_step, 'bid_size6'],
            self.df.loc[self.current_step, 'bid_size7'],
            self.df.loc[self.current_step, 'bid_size8'],
            self.df.loc[self.current_step, 'bid_size9'],
            self.df.loc[self.current_step, 'bid_size10'],
        ], dtype=float)
        return obs

    def _take_action(self, actions):
        # clear your current limit orders
        if len(self.lob_record.keys()) != 0:
            for set_price in self.lob_record.keys():
                # if size > 0 it is buy limit order then we check a1
                if self.lob_record[set_price] > 0 and set_price > self.df.loc[self.current_step, "ask_price1"]:
                    # execute buy one!
                    self.balance -= self.df.loc[self.current_step, "ask_price1"] * self.lob_record[set_price]
                    self.shares_held += self.lob_record[set_price]
                    self.lob_record[set_price] = 0
                    # print(f"{set_price} buy limit order executed!")
                    # print(f"c ask1: {self.df.loc[self.current_step, 'ask_price1']}")

                elif self.lob_record[set_price] < 0 and set_price < self.df.loc[self.current_step, "bid_price1"]:
                    # execute sell one!
                    self.balance += self.df.loc[self.current_step, "bid_price1"] * -self.lob_record[set_price]
                    self.shares_held -= -self.lob_record[set_price]
                    self.lob_record[set_price] = 0
                    # print(f"{set_price} sell limit order executed!")
                    # print(f"current bid1: {self.df.loc[self.current_step, 'bid_price1']}")

        # Set the current price
        mid_price = (self.df.loc[self.current_step, "ask_price1"] + self.df.loc[self.current_step, "bid_price1"]) / 2

        for action in actions:
            action_type = action[0]
            amount = action[1]

            # Put one limit bid order at price a (buy stock)
            if action_type == 0:
                # check it is executed?
                if amount >= self.df.loc[self.current_step, "ask_price1"]:
                    # transaction executed
                    self.balance -= self.df.loc[self.current_step, "ask_price1"]
                    self.shares_held += 1
                else:
                    self.lob_record[amount] += 1


            # Put one limit ask order at price a (sell stock)
            elif action_type == 1:
                # check it is executed?
                if amount <= self.df.loc[self.current_step, "bid_price1"]:
                    # transaction executed
                    self.balance += self.df.loc[self.current_step, "bid_price1"]
                    self.shares_held -= 1
                else:
                    self.lob_record[amount] -= 1

            elif action_type == 2:
                # clear your inventory
                self.balance += self.shares_held * mid_price
                self.shares_held = 0

            self.net_worth = self.balance + self.shares_held * mid_price

            if self.net_worth > self.max_net_worth:
                self.max_net_worth = self.net_worth

            if self.shares_held == 0:
                self.cost_basis = 0

    def step(self, current_action):
        # Execute one time step within the environment
        previous_net_worth = self.net_worth
        self._take_action(current_action)
        done = False

        self.current_step += 1

        if self.current_step > self.len - 1:
            # self.current_step = 0  # loop training
            done = True

        # delay_modifier = (self.current_step / MAX_STEPS)

        # profits
        # reward = self.net_worth - INITIAL_ACCOUNT_BALANCE
        # reward = 1 if reward > 0 else -100
        profit = self.net_worth - self.INITIAL_ACCOUNT_BALANCE
        reward = self.net_worth - previous_net_worth
        if self.net_worth < 0:
            done = True
        obs = self._next_observation()
        if self.current_step == self.len - 1:
            done = True
        return obs, reward, done, {"profit": profit, "inv": self.shares_held, "net_worth": self.net_worth, 'balance':self.balance ,
                                   'reward': reward}

    def reset(self, new_df=None):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.lob_record = defaultdict(int)
        self.current_step = 0

        # pass test dataset to environment

        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(
        #     0, len(self.df.loc[:, 'open'].values) - 6)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - self.INITIAL_ACCOUNT_BALANCE
        real = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print('-' * 30)
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        # print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
        print(f'real_profit:{real}')
        return profit





