import pandas as pd
import torch
from DQN import PolicyGradientAgent
from signal_functions import volatility, transition_times, rsi_increase, AO, MACD, Aroon
from stock_simulator.main import StockTradingEnv

profit_all = []
for t in range(1, 31):
    # t0 is used for select model that is trained by the previous day
    if t == 1:
        t0 = '31'
    else:
        t0 = '0' + str(t - 1) if t < 10 else t

    t = '0' + str(t) if t < 10 else t
    # read the data by day.
    url = '/Users/ming/Documents/Python_project/Project/data/LOB/BTCUSDT_S_DEPTH_202111{0}.csv'.format(t)
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
                'ap10', 'ap10']:
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
    data_len = df1.shape[0]  # data size
    # print(data_len)

    # create RL agent
    best_agent = PolicyGradientAgent(learning_rate=0.0001, input_dims=[13], GAMMA=0.99, n_actions=6,
                                     layer1_size=256, layer2_size=256)
    # initialize the sequence length
    seq_len = 2000

    # load my RL agent
    best_agent.policy.load_state_dict(torch.load("test_{0}.pt".format(t0), map_location='cpu'))

    # create testing set.
    df_test = pd.DataFrame(df1)

    for idx in range(int(data_len / seq_len)):
        # create a period
        df_sample = df1.iloc[idx * seq_len:(idx + 1) * seq_len, :]
        df_sample = df_sample.reset_index(drop=True)
        env = StockTradingEnv(df_sample)  # create the stock trading environment
        observation = env.reset(df_sample)  # read the data from the trading environment
        a10, a9, a8, a7, a6, a5, a4, a3, a2, a1, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, \
        a10s, a9s, a8s, a7s, a6s, a5s, a4s, a3s, a2s, a1s, b1s, b2s, b3s, b4s, b5s, b6s, b7s, b8s, b9s, b10s = observation

        # setting parameters
        score = 0
        done = False
        profit = 0
        i = 0
        p = []
        inv = []
        while not done:
            # observing time
            if i <= 200:
                mid_price = (a1 + b1) / 2
                i += 1
                p.append(mid_price)
                action_space = []
                observation_, reward, done, info = env.step(action_space)
                observation = observation_
                inv.append(info['inv'])
            # trading time
            if i > 200:
                a10, a9, a8, a7, a6, a5, a4, a3, a2, a1, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, \
                a10s, a9s, a8s, a7s, a6s, a5s, a4s, a3s, a2s, a1s, b1s, b2s, b3s, b4s, b5s, b6s, b7s, b8s, b9s, b10s = observation
                mid_price = (a1 + b1) / 2
                p.append(mid_price)
                p = p[-100:]
                vol = volatility(p)
                vol_10 = volatility(p[-10:])
                vol_20 = volatility(p[-20:])
                vol_50 = volatility(p[-50:])
                share_held = inv[-1]
                tt = transition_times(p)
                rsi_in_10 = rsi_increase(p[-10:])
                rsi_in_50 = rsi_increase(p[-50:])
                rsi_in_20 = rsi_increase(p[-20:])
                rsi_in_100 = rsi_increase(p[-100:])
                inventory = inv[-1]
                ao = AO(p)
                macd = MACD(p)
                aroon = Aroon(p[-30:])
                aroon_20 = Aroon(p[-20:])
                aroon_50 = Aroon(p[-50:])

                # create the state
                state = [share_held, rsi_in_10, ao, vol_20, aroon_20, a2, a1, b1, b2, a2s, a1s, b1s, b2s]

                # agent return the action.
                a = best_agent.choose_action(state)
                if a == 0:
                    action_space = [(0, b1), (1, a1)]
                else:
                    action_space = []

                # agent do the action.
                observation_, reward, done, info = env.step(action_space)

                # agent store the rewards for optimizing.
                best_agent.store_rewards(reward)

                # score is accumulative profit.
                score += reward
                inv.append(info['inv'])
                cash = info['balance']

                # loop for the next observation
                observation = observation_
        print('score:', score, 'idx', idx)
        profit_all.append(score)

    # print the total profit
    print(sum(profit_all) * 100000)
