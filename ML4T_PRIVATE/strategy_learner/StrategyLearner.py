# Nick Lee - nlee68

import datetime as dt
import pandas as pd
import util as ut
import OptimizeLearner
import time
import marketsimcode as market


def author():
    return 'nlee68'  # replace tb34 with your Georgia Tech username

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.learner = None #placeholder for agent
        self.classic_orders = None

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        start = time.time()

        # add your code to do learning here
        my_optimize_learner = OptimizeLearner.OptimizeLearner(symbol=symbol,
                                                              sd=sd,ed=ed, commission = 0, impact = self.impact, sv=100000, verbose=False)

        my_optimize_learner.optimize_me()

        self.learner = my_optimize_learner #assign it to the class

        if self.verbose:
            print(self.learner)
            print("Add Evidence took ", time.time()-start)


    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 100000):

        if self.verbose:
            print("The dates I am given are ", sd, " and ", ed)
        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        # trades.values[:,:] = 0 # set them all to nothing

        orders_df = self.learner.run_without_training(sd=sd, ed=ed, sv=sv)
        self.classic_orders = self.learner.get_classic_orders()
        #print orders_df

        # trades.values[0,:] = 1000 # add a BUY at the start
        # trades.values[40,:] = -1000 # add a SELL
        # trades.values[41,:] = 1000 # add a BUY
        # trades.values[60,:] = -2000 # go short from long
        # trades.values[61,:] = 2000 # go long from short
        # trades.values[-1,:] = -1000 #exit on the last day
        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        return orders_df
        #return trades

    def get_classic_policy_for_testing(self):
        #print "birds", self.learner.variables
        return self.classic_orders



if __name__=="__main__":
    #in sample

    strat_learner = StrategyLearner(verbose=False, impact=0)
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)

    start = time.time()
    strat_learner.addEvidence(symbol = "JPM", sd=start_date, ed=end_date, sv=100000)
    print("Add Evidence took %f", time.time()-start)
    orders_df = strat_learner.testPolicy(sd=start_date, ed=end_date, sv=100000)
    orders_df = strat_learner.get_classic_policy_for_testing()
    #benchmark_policy = ind.benchmark_policy(sd=start_date, ed=end_date)

    #benchmark_policy.to_csv('benchmark.csv')
    orders_df.to_csv('orders.csv')

    #print orders_df
    #portvals_raw = compute_portvals_abridged(orders_df, commission=0, impact=0)
    portvals_raw = market.compute_portvals_abridged(orders_df, commission=0, impact=0)
    #benchmark_portvals_raw = compute_portvals_abridged(benchmark_policy, commission=0, impact=0)

    #clean up
    portvals = market.extract_portvals_only(portvals_raw)
    #benchmark_portvals = extract_portvals_only(benchmark_portvals_raw)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = market.compute_portval_stats(portvals, rfr=0.0, sf=252, sv=100000)

 # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)

    print
    print "Cumulative Return of Fund: {}".format(cum_ret)

    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)

    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)

    print
    print "Final Portfolio Value: {}".format(portvals[-1])

