import pandas as pd


def combine(my_data_buy, my_data_sell, added_data):
    """
    this function combine 2 data list which records the price and volume in dictionary data type.
    """
    for tup in added_data["sell"]:
        my_data_sell[tup[0]] += tup[1]
    for tup in added_data["buy"]:
        my_data_buy[tup[0]] += tup[1]

    return my_data_buy, my_data_sell


def plot_bar(my_data_buy, my_data_sell, range_, num_interval, mid_price):
    """
    this function is to create buy and sell histogram.
    """
    buy = pd.DataFrame({'qty': my_data_buy})  # load the data in dataframe
    sell = pd.DataFrame({'qty': my_data_sell}) # load the data in dataframe
    buy = buy.reset_index()
    sell = sell.reset_index()
    buy.columns = ['col1', 'col2']
    sell.columns = ['col1', 'col2']
    buy = buy.sort_values(by='col1')
    sell = sell.sort_values(by='col1')
    delta = range_ / num_interval
    max_value = mid_price + range_
    min_value = mid_price - range_
    buy_ans = buy[(buy['col1'] >= min_value) & (buy['col1'] <= max_value)]
    sell_ans = sell[(sell['col1'] >= min_value) & (sell['col1'] <= max_value)]
    sell_ans = sell_ans.reset_index()
    buy_ans = buy_ans.reset_index()
    for i in range(sell_ans.shape[0]):
        for price in [mid_price - i * delta for i in range(num_interval, 0, -1)] + \
                     [mid_price + i * delta for i in range(num_interval)]:
            if sell_ans.loc[i, 'col1'] >= price and (sell_ans.loc[i, 'col1'] < price + delta):
                sell_ans.loc[i, 'price'] = price
                break
    for i in range(buy_ans.shape[0]):
        for price in [mid_price - i * delta for i in range(num_interval, 0, -1)] + \
                     [mid_price + i * delta for i in range(num_interval)]:
            if buy_ans.loc[i, 'col1'] >= price and (buy_ans.loc[i, 'col1'] < price + delta):
                buy_ans.loc[i, 'price'] = price
                break
    df = pd.DataFrame([mid_price - i * delta for i in range(num_interval, 0, -1)] +
                      [mid_price + i * delta for i in range(num_interval)])
    df.columns = ['price']
    if 'price' not in buy_ans.columns:
        buy_ans = pd.DataFrame([mid_price - i * delta for i in range(num_interval, 0, -1)] +
                               [mid_price + i * delta for i in range(num_interval)])
        buy_ans.columns = ['price']
        buy_ans['col2'] = 0
    else:
        buy_ans = buy_ans[['col2', 'price']].groupby(['price'], as_index=False).sum()
        buy_ans = df.merge(buy_ans, how='left', on='price')
        buy_ans = buy_ans.fillna(0)

    if 'price' not in sell_ans.columns:
        sell_ans = pd.DataFrame([mid_price - i * delta for i in range(num_interval, 0, -1)] +
                                [mid_price + i * delta for i in range(num_interval)])
        sell_ans.columns = ['price']
        sell_ans['col2'] = 0
    else:
        sell_ans = sell_ans[['col2', 'price']].groupby(['price'], as_index=False).sum()
        sell_ans = df.merge(sell_ans, how='left', on='price')
        sell_ans = sell_ans.fillna(0)

    return buy_ans, sell_ans
