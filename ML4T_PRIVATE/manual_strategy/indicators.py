#this should take in dataframes
import datetime as dt
import marketsimcode as market
import pandas as pd
import matplotlib.pyplot as plt
import BestPossibleStrategy as bps

def get_bollinger(symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31)):
    bollinger_df = market.make_df_to_match_trading_days(colnames=['Date', 'bb1', 'bb2'], symbol=symbol, sd=sd, ed=ed)
    SMA_df = SMA(symbol, sd=sd, ed=ed)
    STD_rolling = rolling_STD(symbol, sd=sd, ed=ed)
    bollinger_df['bb1'] = SMA_df['SMA'] + (STD_rolling['STD20'] * 2)
    bollinger_df['bb2'] = SMA_df['SMA'] - (STD_rolling['STD20'] * 2)
    return bollinger_df

def SMA(symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31)):
    sd_left_bracket = sd - dt.timedelta(days=40) #we need to go back in time to get data for SMA. #40 days is conservative
    SMA_df = market.make_df_to_match_trading_days(colnames=['Date', 'SMA'], symbol='JPM', sd=sd_left_bracket, ed=ed)
    stock_price_library = market.get_stock_prices([symbol], sd_left_bracket, ed)

    SMA_calcs = stock_price_library.rolling(window=20, min_periods=20, center=False).mean()
    SMA_df['SMA'] = SMA_calcs[symbol]
    SMA_df['P/SMA'] = stock_price_library[symbol]/SMA_calcs[symbol]

    #print sorted(SMA_df[sd:ed]['P/SMA'])

    return SMA_df[sd:ed] #returns the 20 day rolling mean for the symbol you feed it

def rolling_STD(symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31)):
    sd_left_bracket = sd - dt.timedelta(days=40) #we need to go back in time to get data for SMA. #40 days is conservative
    STD_df = market.make_df_to_match_trading_days(colnames=['Date', 'STD20'], symbol='JPM', sd=sd_left_bracket, ed=ed)
    stock_price_library = market.get_stock_prices([symbol], sd_left_bracket, ed)

    STD_calcs = stock_price_library.rolling(window=20, min_periods=20, center=False).std()
    STD_df['STD20'] = STD_calcs[symbol]

    return STD_df[sd:ed] #returns the 20 day rolling mean for the symbol you feed it

def stochastic_oscillator_20(symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31)):
    sd_left_bracket = sd - dt.timedelta(days=40)
    STOK_df = market.make_df_to_match_trading_days(colnames=['Date', '%K'], symbol='JPM', sd=sd_left_bracket, ed=ed)
    stock_price_library = market.get_stock_close_nonadjusted([symbol], sd_left_bracket, ed)

    low_df = market.get_stock_lows([symbol], sd_left_bracket, ed)
    high_df = market.get_stock_highs([symbol], sd_left_bracket, ed)

    STOK_df['%K'] = ((stock_price_library - low_df.rolling(window=14, center=False).min()) /
                     (high_df.rolling(window=14, center=False).max() - low_df.rolling(window=14, center=False).min())) * 100

    STOK_df['%D'] = STOK_df.rolling(window=3, center=False).mean()
    return STOK_df[sd:ed]

def plot_prices(symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31)):
    stock_price_library = market.get_stock_prices([symbol], sd, ed)
    #print stock_price_library

    price_line = plt.plot(stock_price_library.index, stock_price_library[symbol], label=symbol)
    plt.setp(price_line, color='r', linewidth=1.0)

    plt.xlabel("Date")
    plt.ylabel(("Stock Price"))
    plt.title("Stock Price vs Date")

    return #fig #generate some plots

def plot_SMA(symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31)):

    SMA_df = SMA(symbol, sd=sd, ed=ed)
    sma_line = plt.plot(SMA_df.index, SMA_df['SMA'], label="SMA-JPM")
    plt.setp(sma_line, linestyle='--', color='b', linewidth=1.0)

def plot_bollinger(symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31)):
    bollinger_df = get_bollinger(symbol, sd=sd, ed=ed)
    bb1 = plt.plot(bollinger_df.index, bollinger_df['bb1'], label='Bollinger Bands')
    bb2 = plt.plot(bollinger_df.index, bollinger_df['bb2'])
    plt.setp(bb1, color='green', linewidth=1.0, linestyle='-')
    plt.setp(bb2, color='green', linewidth=1.0, linestyle='-')

def plot_STOK(symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), incl_STOD=True):
    STOK_df = stochastic_oscillator_20(symbol, sd=sd, ed=ed)
    K = plt.plot(STOK_df.index, STOK_df['%K'], label="%K")
    if incl_STOD is True:
        D = plt.plot(STOK_df.index, STOK_df['%D'], label='%D')
        plt.setp(D, color='blue', linewidth=1.0, linestyle='-')
    plt.setp(K, color='green', linewidth=1.0, linestyle='-')


#SMA('JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31))

def benchmark_policy(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
    orders_df = market.make_df_to_match_trading_days(colnames=['Date','Symbol', 'Order', 'Shares'],
                                                     symbol='JPM', sd=sd, ed=ed)

    orders_df['Symbol'] = symbol #assign to JPM
    orders_df['Order'] = "" #initialize with no orders
    orders_df['Shares'] = 0.0 #initialize with no shares
    orders_df['Date'] = orders_df.index

    first_order_date = orders_df.iloc[0]['Date']
    orders_df.set_value(first_order_date, 'Order', 'BUY')
    orders_df.set_value(first_order_date, 'Shares', 1000)

    #print orders_df

    return orders_df


if __name__ == "__main__":

    #show bollinger and SMA on stock price
    fig = plt.figure()

    #in sample
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)

    #out of sample
    # start_date = dt.datetime(2010,1,1)
    # end_date = dt.datetime(2011, 12, 31)

    plot_prices(symbol='ML4T-220',sd=start_date, ed=end_date)
    plot_SMA(symbol='ML4T-220', sd=start_date, ed=end_date)
    plot_bollinger(symbol='ML4T-220', sd=start_date, ed=end_date)
    plot_STOK(symbol='ML4T-220', sd=start_date, ed=end_date)
    legend = plt.legend(loc='best', shadow=True)

    plt.grid()
    plt.savefig("final.png")
    plt.show()
    #
    # fig2 = plt.figure(2)
    # plot_STOK()
    # plt.legend(loc='best', shadow=True)
    # plt.show()

    #show STOKD and JP Morgan subplots
    plt.figure(1)
    plt.subplot(211)
    plot_STOK(symbol='ML4T-220',sd=start_date, ed=end_date)
    plt.title("Stochastic Oscillator vs Time for JPY")
    plt.legend(loc='best', shadow=True)

    plt.subplot(212)
    plot_prices(symbol='ML4T-220')
    plt.show()



