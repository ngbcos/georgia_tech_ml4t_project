"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import scipy.optimize as spo
from util import get_data, plot_data
import fortune500


#HELPER FUNCTIONS

def normalize_data(df):
    """Normalize stock prices with first row. Code borrowed from lecture."""
    return df/ df.ix[0,:]

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

def minimize_this_function(allocs, prices):
    rfr = 0.0
    sf = 252.0
    sv = 10^6
    cr, adr, sddr, sr = compute_portfolio_stats(prices, allocs, rfr, sf, sv)
    return float(1)/sr

#END HELPER FUNCTIONS

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    #initial guess
    allocs = np.zeros(len(syms))
    allocs.fill(1/float(len(syms))) #initial guess is an even distribution

    #minimize_this_function(allocs, prices)

    #run code for optimization
    bnds = tuple((0,1) for stock in allocs)
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)}) #code borrowed from StackOverflow guarantees all elements sum to 1
    #https://stackoverflow.com/questions/18767657/how-do-i-use-a-minimization-function-in-scipy-with-constraints

    res = spo.minimize(minimize_this_function, allocs, args=(prices,), method = 'SLSQP', constraints=cons, bounds=bnds)

    allocs = res.x

    cr, adr, sddr, sr = compute_portfolio_stats(prices, allocs, 0, 252.0, 10^6)

    # Get daily portfolio value
    value_by_day_normalized = allocs * normalize_data(prices)  # tells us how each much day is worth
    total_value_by_day = value_by_day_normalized.sum(axis=1)
    port_val = total_value_by_day # computes daily portfolio values (normalized)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        prices_SPY_normalized = normalize_data(prices_SPY)
        df_temp = pd.concat([port_val, prices_SPY_normalized], keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp, title = "Optimized Portfolio and SPY value (normalized) vs. Time", ylabel="Normalized Price")
        pass

    return allocs, cr, adr, sddr, sr

def print_sort_allocs(symbols, allocs, prices):
    alloc_dictionary = dict(zip(symbols, allocs))
    money_dictionary = {}

    for stock in alloc_dictionary:
        money_dictionary[stock] = alloc_dictionary[stock] * float(10992) / prices.ix[-1, stock] #this should tell us how many shares, approx we need

    print(sorted(alloc_dictionary.items(), key=lambda p:p[1], reverse=True))
    print(sorted(money_dictionary.items(), key=lambda p:p[1], reverse=True))


    #print(sorted_dictionary) #a simple stupid command to print the sorted dict



def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2016,8,24)
    end_date = dt.datetime(2018,8,24)
    symbols = fortune500.nicks_other_fav_dividend_stocks
    symbols = fortune500.dividend_stocks + fortune500.tech_winners + fortune500.nicks_other_fav_dividend_stocks #fortune500.dividend_stocks + fortune500.tech_winners #['LVS','ETP','MSFT','IBM', 'DUK', 'KO', 'SDT', 'GOOG', 'AMZN', 'AIG']
    #fortune500.fortune_500
    #['LVS','ETP','MSFT','IBM', 'DUK', 'KO', 'SDT']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr


    #this is hacky code but whatever

    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols


    print_sort_allocs(symbols, allocations, prices)

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
