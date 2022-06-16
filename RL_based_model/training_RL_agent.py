import numpy as np
import pandas as pd
import torch

from DQN import PolicyGradientAgent
from stock_simulator.main import StockTradingEnv
from signal_functions import volatility, rsi_increase, AO, MACD, Aroon

if __name__ == '__main__':
    for t in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
              "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
              "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]:
        if t == '31':
            url = '../data/LOB/BTCUSDT_S_DEPTH_202110{0}.csv'.format(t)
        else:
            url = '../data/LOB/BTCUSDT_S_DEPTH_202111{0}.csv'.format(t)
        # read the data.
        data = pd.read_csv(url)

        # select the columns that we need.
        df1 = data[['ap1', 'as1', 'bp1', 'bs1',
                    'ap2', 'as2', 'bp2', 'bs2',
                    'ap3', 'as3', 'bp3', 'bs3',
                    'ap4', 'as4', 'bp4', 'bs4',
                    'ap5', 'as5', 'bp5', 'bs5',
                    'ap6', 'as6', 'bp6', 'bs6',
                    'ap7', 'as7', 'bp7', 'bs7',
                    'ap8', 'as8', 'bp8', 'bs8',
                    'ap9', 'as9', 'bp9', 'bs9',
                    'ap10', 'as10', 'bp10', 'bs10']]

        # choose the price data for normalizing.
        price_data = df1[['ap1', 'bp1',
                          'ap2', 'bp2',
                          'ap3', 'bp3',
                          'ap4', 'bp4',
                          'ap5', 'bp5',
                          'ap6', 'bp6',
                          'ap7', 'bp7',
                          'ap8', 'bp8',
                          'ap9', 'bp9',
                          'ap10', 'bp10']]

        # normalize the price.
        for col in ['ap1', 'bp1',
                    'ap2', 'bp2',
                    'ap3', 'bp3',
                    'ap4', 'bp4',
                    'ap5', 'bp5',
                    'ap6', 'bp6',
                    'ap7', 'bp7',
                    'ap8', 'bp8',
                    'ap9', 'bp9',
                    'ap10', 'bp10']:
            df1[col] = df1[col] / 100000

        # set columns
        columns = ['ask_price1', 'ask_size1', 'bid_price1', 'bid_size1',
                   'ask_price2', 'ask_size2', 'bid_price2', 'bid_size2',
                   'ask_price3', 'ask_size3', 'bid_price3', 'bid_size3',
                   'ask_price4', 'ask_size4', 'bid_price4', 'bid_size4',
                   'ask_price5', 'ask_size5', 'bid_price5', 'bid_size5',
                   'ask_price6', 'ask_size6', 'bid_price6', 'bid_size6',
                   'ask_price7', 'ask_size7', 'bid_price7', 'bid_size7',
                   'ask_price8', 'ask_size8', 'bid_price8', 'bid_size8',
                   'ask_price9', 'ask_size9', 'bid_price9', 'bid_size9',
                   'ask_price10', 'ask_size10', 'bid_price10', 'bid_size10'
                   ]

        df1.columns = columns
        data_len = df1.shape[0]

        # print data length
        # print(data_len)
        df_train = df1
        print('day:', t)

        # print data head
        # print(df_train.head())

        # create RL agent
        agent = PolicyGradientAgent(learning_rate=0.001, input_dims=[13], GAMMA=0.99, n_actions=4,
                                    layer1_size=256, layer2_size=256)

        # initialize the score, number of episodes, sequence length of each , batch size
        score = 0
        num_episodes = 20
        seq_len = 2000
        batch_size = 75

        # start training
        for k in range(num_episodes):
            # record the profit.
            profit_all = []

            for j in range(batch_size):
                # randomly choose the one training sample
                # cur_df = df1.iloc[j * seq_len:(j + 1) * seq_len, :]
                sample_idx = np.random.randint(0, data_len - seq_len, 1)[0]
                cur_df = df_train.iloc[sample_idx:sample_idx + seq_len, :]
                cur_df = cur_df.reset_index(drop=True)

                # create the stock trading environment
                env = StockTradingEnv(cur_df)

                # read the data
                observation = env.reset(cur_df)
                a10, a9, a8, a7, a6, a5, a4, a3, a2, a1, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, \
                a10s, a9s, a8s, a7s, a6s, a5s, a4s, a3s, a2s, a1s, b1s, b2s, b3s, b4s, b5s, b6s, b7s, b8s, b9s, b10s = observation

                # setting parameters
                done = False  # done is true means the trading ends.
                score = 0  # score records the profit after each step
                num = 0  # num control the time that the agent observe.
                p = []  # P records the mid price.
                inv = []  # inv records the inventory.
                while not done:
                    # observe time
                    if num < 100:
                        mid_price = (a1 + b1) / 2
                        num += 1
                        p.append(mid_price)
                        observation_, reward, done, info = env.step([(3, 0)])
                        inv.append(info['inv'])
                        cash = info['balance']
                        observation = observation_

                    # trading time
                    if num >= 100:
                        mid_price = (a1 + b1) / 2
                        p.append(mid_price)
                        p = p[-100:]

                        # calculate signals.
                        vol = volatility(p)
                        vol_10 = volatility(p[-10:])
                        vol_20 = volatility(p[-20:])
                        vol_50 = volatility(p[-50:])
                        share_held = inv[-1]
                        rsi_in_10 = rsi_increase(p[-10:])
                        rsi_in_50 = rsi_increase(p[-50:])
                        rsi_in_100 = rsi_increase(p)
                        inventory = inv[-1]
                        ao = AO(p)
                        macd = MACD(p)
                        aroon = Aroon(p[-30:])
                        aroon_20 = Aroon(p[-20:])

                        # create the state
                        state = [share_held, rsi_in_10, ao, vol_20, aroon_20, a2, a1, b1, b2, a2s, a1s, b1s, b2s]

                        # agent return the action.
                        a = agent.choose_action(state)
                        action_space = []
                        if a == 0:
                            # put limit orders on the top of LOB
                            action_space = [(0, b1), (1, a1)]
                        else:
                            # hold and do nothing
                            action_space = []

                        # agent do the action.
                        observation_, reward, done, info = env.step(action_space)

                        # agent store the rewards for optimizing.
                        agent.store_rewards(reward)

                        # score is accumulative profit.
                        score += reward

                        # store the inventory, and
                        inv.append(info['inv'])

                        # loop for the next observation
                        observation = observation_
                # agent optimizing the parameters in policy making network.
                agent.learn()

                # records the profit after each sample.
                profit_all.append(score)

                # print the final return after each sample
                print('episode: ', k + 1, 'bs:', j, ' score: ', score)

            # print profit rate after one batch.
            print(f'profit_rate: {sum([1 if i >= 0 else 0 for i in profit_all]) / len(profit_all)} ')
            # print the total profit
            print('made money:', sum(profit_all) * 100000)
            # if sum(profit_all)*100000 > 3000:
            #     break

        # store our model agent by date.
        # torch.save(agent.policy.state_dict(), "test_{0}.pt".format(t))
