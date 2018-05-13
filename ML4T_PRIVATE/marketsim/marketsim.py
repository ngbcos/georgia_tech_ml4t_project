"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def calculate_trade_deductions(order_type, share_price, share_count, commission = 9.95, impact = 0.005):
    deduction = commission + share_price*impact*share_count
    #print "Ouch you lost ", deduction
    return deduction #is order type irrelevant?

def get_stock_price(symbols, start_date, end_date):
    stock_prices = get_data(symbols, pd.date_range(start_date, end_date))
    return stock_prices

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
                if order == 'BUY':
                    current_stocks_dictionary[symbol] += shares
                    cash = cash - stock_price*shares
            else:
                if order == 'SELL':
                    current_stocks_dictionary[symbol] = -shares
                    cash = cash+stock_price*shares
                if order == 'BUY':
                    current_stocks_dictionary[symbol] = shares
                    cash = cash-stock_price*shares

            #deduct cash from the commissions and impact
            cash = cash - calculate_trade_deductions(order, stock_price, shares, commission, impact)

            #Recalculate portfolio value with fees and commissions
            #print current_stocks_dictionary
            current_port_value = cash + value_my_shares(current_stocks_dictionary, stock_price_library, order_date)

            current_values_dictionary[str(order_date)] = current_port_value #update the current value in the dictionary
    return current_stocks_dictionary, current_values_dictionary, cash

def author():
    return 'nlee68'  # replace tb34 with your Georgia Tech username

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
    stock_price_library = get_stock_price(list_of_symbols, start_date, end_date)
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

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    script_dir = os.path.dirname(__file__)
    rel_path_train = "orders/orders-01.csv"
    abs_file_path = os.path.join(script_dir, rel_path_train)
    print abs_file_path

    of = abs_file_path
    #pd.read_csv(of, index_col='Date', parse_dates=True, usecols=['Date', 'Symbol', 'Order', 'Shares'], na_values=['nan'])
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
