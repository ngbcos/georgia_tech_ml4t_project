"""Analyze a portfolio."""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data


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

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
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

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010,06,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr
    print "End Value", ev

if __name__ == "__main__":
    test_code()

"""Start Date: 2010-01-01
End Date: 2010-12-31
Symbols: ['GOOG', 'AAPL', 'GLD', 'XOM']
Allocations: [0.2, 0.3, 0.4, 0.1]
Sharpe Ratio: 1.51819243641
Volatility (stdev of daily returns): 0.0100104028
Average Daily Return: 0.000957366234238
Cumulative Return: 0.255646784534"""