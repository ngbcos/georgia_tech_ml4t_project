"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import BestPossibleStrategy as bps
import ManualStrategy as ms
import matplotlib.pyplot as plt
import indicators as ind

#legacy code

def compute_portfolio_stats(prices, allocs, rfr, sf, sv):

    #print allocs

    #normalize the prices according to first day
    prices_normalized = normalize_data(prices)
    #print prices_normalized

    #calculate CR
    last_row_normalized = prices_normalized.ix[-1,:]
    #print last_row_normalized
    multiplier = sum(allocs*last_row_normalized) #This would be 1.37 for a 37% return
    #print multiplier
    cr = multiplier - 1 #subtract 1 because we are interested in the difference from start

    #daily returns
    value_by_day = allocs*prices_normalized*sv #tells us how each much day is worth
    total_value_by_day = value_by_day.sum(axis=1)

    #cumulative_return_by_day = return_by_day.sum(axis=1) #just for fun
    #print cumulative_return_by_day

    #using total value by day should give me my daily returns
    daily_returns = total_value_by_day.copy()
    daily_returns[1:] = (total_value_by_day[1:] / total_value_by_day[:-1].values) - 1 #code borrowed from lecture
    daily_returns.ix[0] = 0 #including this for thoroughness

    daily_returns = daily_returns[1:] #drop row 1

    #print daily_returns
    adr = daily_returns.mean()

    sddr = daily_returns.std()

    #back calculate daily risk free rate given the annual risk free rate
    daily_rfr = (1+rfr)**(1/float(252)) - 1

    sr = (daily_returns - daily_rfr).mean() / (daily_returns - daily_rfr).std()

    #Convert to annualized sharpe ratio
    K = sf**(1/float(2))
    sr = K * sr

    return cr, adr, sddr, sr

def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    port_val = prices_SPY # add code here to compute daily portfolio values

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = \
        compute_portfolio_stats(prices=prices, \
                                allocs=allocs, \
                                rfr=rfr, sf=sf, sv = sv)


    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        pass

    ev = sv + sv * cr  # formula for end value of portfolio

    return cr, adr, sddr, sr, ev


def compute_portval_stats(portvals, rfr=0.0, sf=252, sv=100000):

    cr = portvals[-1]/float(sv) - 1

    #daily returns
    total_value_by_day = portvals

    #cumulative_return_by_day = return_by_day.sum(axis=1) #just for fun
    #print cumulative_return_by_day

    #using total value by day should give me my daily returns
    daily_returns = total_value_by_day.copy()
    daily_returns[1:] = (total_value_by_day[1:] / total_value_by_day[:-1].values) - 1 #code borrowed from lecture
    daily_returns.ix[0] = 0 #including this for thoroughness

    daily_returns = daily_returns[1:] #drop row 1

    #print daily_returns
    adr = daily_returns.mean()

    sddr = daily_returns.std()

    #back calculate daily risk free rate given the annual risk free rate
    daily_rfr = (1+rfr)**(1/float(252)) - 1

    sr = (daily_returns - daily_rfr).mean() / (daily_returns - daily_rfr).std()

    #Convert to annualized sharpe ratio
    K = sf**(1/float(2))
    sr = K * sr

    return cr, adr, sddr, sr

def normalize_data(df):
    """Normalize stock prices with first row. Code borrowed from lecture."""
    return df/ df.ix[0,:]


def calculate_trade_deductions(order_type, share_price, share_count, commission = 9.95, impact = 0.005):
    deduction = commission + share_price*impact*share_count
    #print "Ouch you lost ", deduction
    return deduction #is order type irrelevant?

def get_stock_prices(symbols, start_date, end_date):
    stock_prices = get_data(symbols, pd.date_range(start_date, end_date))
    return stock_prices

def get_stock_lows(symbols, start_date, end_date):
    stock_lows = get_data(symbols, pd.date_range(start_date, end_date), colname='Low')
    return stock_lows

def get_stock_highs(symbols, start_date, end_date):
    stock_highs = get_data(symbols, pd.date_range(start_date, end_date), colname='High')
    return stock_highs

def get_stock_close_nonadjusted(symbols, start_date, end_date):
    stock_closes = get_data(symbols, pd.date_range(start_date, end_date), colname='Close')
    return stock_closes

def make_df_to_match_trading_days(colnames=['Date', 'Value'], symbol='JPM',
                                  sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31)):

    stock_price_library = get_stock_prices([symbol], sd, ed)
    generic_df = pd.DataFrame(columns=colnames)
    generic_df['Date'] = stock_price_library.index.copy()
    generic_df.set_index('Date', inplace=True)

    return generic_df

def get_stock_price_on_date(symbol, price_library, date):
    #print "Let me look up ", symbol
    stock_of_interest = price_library[symbol]
    #print stock_of_interest
    return stock_of_interest[date]

def value_my_shares(current_stocks_dictionary, price_library, date):
    #this takes in a dictionary of stock tickers and share amounts
    total_stock_value = 0.0
    for ticker, shares in current_stocks_dictionary.iteritems():
        total_stock_value += shares*get_stock_price_on_date(ticker, price_library, date)
    return total_stock_value

def process_orders_on_date(cash, trading_date, orders_df, stock_price_library, current_stocks_dictionary,
                           current_values_dictionary, commission, impact):
    trading_date = pd.Timestamp(trading_date) #convert trading date to a Pandas thingy

    #this could be made more efficient by extracting out the orders for the dates we care about insted of checking each date
    #But I'm kind of tired right now and I pass all the tests, so I'm going to call it a day

    for row in orders_df.itertuples(index=True):
        # Get order
        order_date = getattr(row,'Date')
        symbol = getattr(row, 'Symbol')
        #print "The symbol is ", symbol
        order = getattr(row, 'Order')
        shares = getattr(row, 'Shares')


        if trading_date != order_date:
            pass

        elif trading_date == order_date:
            #print "I want to ", order, " ", shares, " of ", symbol, " on ", order_date

            #Run the order
            stock_price = get_stock_price_on_date(symbol, stock_price_library, order_date)
            #print "Today the stock ", symbol, " is worth ", stock_price

            #print "zomg", symbol
            #add or remove shares from dictionary of shares
            if symbol in current_stocks_dictionary:
                if order == 'SELL':
                    current_stocks_dictionary[symbol] -= shares
                    cash = cash + stock_price*shares

                    #plotcode
                    if current_stocks_dictionary[symbol] < 0:
                        plt.axvline(order_date, color='r', linestyle='-')  # EFFING BRITTLE CODE

                if order == 'BUY':
                    current_stocks_dictionary[symbol] += shares
                    cash = cash - stock_price*shares

                    #plotcode
                    if current_stocks_dictionary[symbol] > 0:
                        plt.axvline(order_date, color='g', linestyle='-')  # EFFING BRITTLE CODE
            else:
                if order == 'SELL':
                    current_stocks_dictionary[symbol] = -shares
                    cash = cash+stock_price*shares

                    #plotcode
                    if current_stocks_dictionary[symbol] < 0:
                        plt.axvline(order_date, color='r', linestyle='-')  # EFFING BRITTLE CODE

                if order == 'BUY':
                    current_stocks_dictionary[symbol] = shares
                    cash = cash-stock_price*shares

                        # plotcode
                    if current_stocks_dictionary[symbol] > 0:
                        plt.axvline(order_date, color='g', linestyle='-')  # EFFING BRITTLE CODE

            #deduct cash from the commissions and impact
            if shares != 0:
                cash = cash - calculate_trade_deductions(order, stock_price, shares, commission, impact)
            else:
                pass #if there are no shares, then don't calculate a deduction

            #Recalculate portfolio value with fees and commissions
            #print current_stocks_dictionary
            current_port_value = cash + value_my_shares(current_stocks_dictionary, stock_price_library, order_date)

            current_values_dictionary[str(order_date)] = current_port_value #update the current value in the dictionary
    return current_stocks_dictionary, current_values_dictionary, cash

def author():
    return 'nlee68'  # replace tb34 with your Georgia Tech username

    #this method skips the file reading and just takes in an orders df.
def compute_portvals_abridged(orders_df, start_val=100000, commission=0.0, impact=0.0):

    orders_df = orders_df.sort_values(['Date'])
    orders_column = orders_df['Date']
    start_date = orders_column.head(1).dt
    end_date = orders_column.tail(1).dt
    start_date = dt.datetime(start_date.year, start_date.month, start_date.day)
    end_date = dt.datetime(end_date.year, end_date.month, end_date.day)


    list_of_symbols = orders_df.Symbol.unique().tolist() #This really makes me wish Python was typesafe

    stock_price_library = get_stock_prices(list_of_symbols, start_date, end_date)

    # Add a portval on the right
    port_val_ledger = pd.DataFrame(index=stock_price_library.index.copy())
    # print port_val_ledger
    port_val_ledger['Date'] = port_val_ledger.index
    port_val_ledger['portvals'] = np.nan

    cash = start_val
    current_port_value = cash
    current_stocks_dictionary = {}
    current_values_dictionary = {}
    # print orders_df

    list_of_dates = port_val_ledger.index.values

    for date in list_of_dates:
        if (date == orders_df['Date']).any():
            # print ('An order exists on this date so I should run through my orders')
            current_stocks_dictionary, \
            current_values_dictionary, \
            cash = process_orders_on_date(cash, date, orders_df, stock_price_library,
                                          current_stocks_dictionary, current_values_dictionary, commission, impact)
        else:
            # calculate portfolio value like normal
            current_port_value = cash + value_my_shares(current_stocks_dictionary, stock_price_library, date)
            current_values_dictionary[str(date)] = current_port_value  # update the current value in the dictionary

    port_vals_converted = pd.Series(current_values_dictionary, name='PortVal').to_frame()

    return port_vals_converted


#   this code needs to be modified to take in a dataframe!!!! 
def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    #set up portvals dataframe
    portvals = pd.DataFrame(columns=['Date', 'Value'])

    #print type(orders_file)
    if not hasattr(orders_file,'read'):
        #print "setting up the file"
        orders_file = open(orders_file, 'r')
    orders_df = pd.read_csv(orders_file, parse_dates=['Date'],
                            usecols=['Date', 'Symbol', 'Order', 'Shares'], na_values=['nan'])

    #orders_df['Date'] = pd.to_datetime(orders_df['Date'])
    #print orders_df.select_dtypes(include=[np.datetime64])
    #orders_df.set_index('Date', inplace=True)
    #orders_df.sort_index(inplace=True)
    orders_df = orders_df.sort_values(['Date'])
    orders_column = orders_df['Date']
    start_date = orders_column.head(1).dt
    end_date = orders_column.tail(1).dt
    start_date = dt.datetime(start_date.year, start_date.month, start_date.day)
    end_date = dt.datetime(end_date.year, end_date.month, end_date.day)

    #print orders_df
    list_of_symbols = orders_df.Symbol.unique().tolist() #This really makes me wish Python was typesafe

    #print list_of_symbols
    stock_price_library = get_stock_prices(list_of_symbols, start_date, end_date)
    #print stock_price_library

    #Add a portval on the right
    port_val_ledger = pd.DataFrame(index=stock_price_library.index.copy())
    #print port_val_ledger
    port_val_ledger['Date'] = port_val_ledger.index
    port_val_ledger['portvals'] = np.nan

    cash = start_val
    current_port_value = cash
    current_stocks_dictionary = {}
    current_values_dictionary = {}
    #print orders_df

    list_of_dates = port_val_ledger.index.values

    for date in list_of_dates:
        if (date == orders_df['Date']).any():
            #print ('An order exists on this date so I should run through my orders')
            current_stocks_dictionary, \
            current_values_dictionary, \
            cash = process_orders_on_date(cash, date, orders_df, stock_price_library,
                                          current_stocks_dictionary, current_values_dictionary, commission, impact)
        else:
            #calculate portfolio value like normal
            current_port_value = cash + value_my_shares(current_stocks_dictionary, stock_price_library, date)
            current_values_dictionary[str(date)] = current_port_value  # update the current value in the dictionary

    port_vals_converted = pd.Series(current_values_dictionary, name='PortVal').to_frame()

    return port_vals_converted

def extract_portvals_only(portvals_dataframe):
    return portvals_dataframe[portvals_dataframe.columns[0]]

def test_code():

    #in sample
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)

    #out-of-sample dates
    # start_date = dt.datetime(2010,1,1)
    # end_date = dt.datetime(2011, 12, 31)

    #orders_df = bps.testPolicy(sd=start_date, ed=end_date)
    plt.figure() #I'm going to try to cheat

    orders_df = ms.testPolicy(sd=start_date, ed=end_date)
    benchmark_policy = ind.benchmark_policy(sd=start_date, ed=end_date)

    benchmark_policy.to_csv('benchmark.csv')
    orders_df.to_csv('orders.csv')

    #portvals_raw = compute_portvals_abridged(orders_df, commission=0, impact=0)
    portvals_raw = compute_portvals_abridged(orders_df, commission=9.95, impact=0.005)
    benchmark_portvals_raw = compute_portvals_abridged(benchmark_policy, commission=0, impact=0)

    #clean up
    portvals = extract_portvals_only(portvals_raw)
    benchmark_portvals = extract_portvals_only(benchmark_portvals_raw)


    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portval_stats(portvals, rfr=0.0, sf=252, sv=100000)
    bench_cum_ret, bench_avg_daily_ret, bench_std_daily_ret, bench_sharpe_ratio = compute_portval_stats(benchmark_portvals, rfr=0.0, sf=252, sv=100000)


    #Get SPY data
    symbols = ['SPY']
    allocations = [1]
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio of SPY
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY, ev = assess_portfolio(sd = start_date, ed = end_date,syms = symbols,
                                             allocs = allocations,sv = 100000, gen_plot = False)

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of Benchmark: {}".format(bench_sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of Benchmark: {}".format(bench_cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of Benchmark: {}".format(bench_std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of Benchmark: {}".format(bench_avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])
    print "Final Benchmark Value: {}".format(benchmark_portvals[-1])
    print "Final SPY Value: {}".format(ev)

    benchmark_normalized = normalize_data(benchmark_portvals_raw)
    benchmark_normalized = extract_portvals_only(benchmark_normalized)
    best_portfolio_normalized = normalize_data(portvals_raw)
    best_portfolio_normalized = extract_portvals_only(best_portfolio_normalized)

    stock_library = make_df_to_match_trading_days(colnames=['Date', 'Value'], symbol='JPM',
                                  sd=start_date, ed=end_date)


    benchmark_line = plt.plot(stock_library.index, benchmark_normalized.values, label="Benchmark")
    plt.setp(benchmark_line, linestyle='-', color='b', linewidth=1.0)
    best_portfolio_line = plt.plot(stock_library.index, best_portfolio_normalized.values, label="Manual Strategy")
    plt.setp(best_portfolio_line, linestyle='-', color='k', linewidth=1.0)
    legend = plt.legend(loc='best', shadow=True)
    plt.title("Normalized chart for Portfolios")
    plt.ylabel("Normalized Value")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    test_code()
