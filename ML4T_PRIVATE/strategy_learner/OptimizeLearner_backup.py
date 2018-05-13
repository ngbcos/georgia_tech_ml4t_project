"""
Template for implementing Optimize Learner
"""

import numpy as np
import random
import copy
import datetime as dt
import scipy.optimize as spo

import marketsimcode as market
import indicators as indicators
import time


# class MyBounds(object):
#     def __init__(self, xmax=[1.1, 1.1], xmin=[-1.1, -1.1]):
#         self.xmax = np.array(xmax)
#         self.xmin = np.array(xmin)
#
#     def __call__(self, **kwargs):
#         x = kwargs["x_new"]
#         tmax = bool(np.all(x <= self.xmax))
#         tmin = bool(np.all(x >= self.xmin))
#         return tmax and tmin
#
# class MyTakeStep(object):
#     def __init__(self, stepsize=0.5):
#         self.stepsize = stepsize
#
#     def __call__(self, x):
#         s = self.stepsize
#         x[0] += np.random.uniform(-2. * s, 2. * s)
#         x[1:] += np.random.uniform(-s, s, x[1:].shape)
#         return x

class OptimizeLearner(object):
    def __init__(self, \
                 variables = [1, 1, 20, 80, 0.7], \
                 symbol="JPM", \
                 sd=dt.datetime(2008, 1, 1), \
                 ed=dt.datetime(2009, 1, 1),
                 commission = 0,
                 impact = 0.005,
                 sv=100000,
                 verbose=False):

        self.variables = variables
        self.symbol = symbol
        self.sv = sv
        self.sd = sd
        self.ed = ed
        self.verbose = verbose
        self.commission = commission
        self.impact = impact

        self.stock_price_library = None #attempt to pre-cache something

        print "oh boy"

    def minimize_this(self, variables):
        #print variables
        orders_df = self.testPolicy(symbol=self.symbol, sd=self.sd, ed=self.ed, sv=self.sv, variables=variables)

        #attempting to make this faster
        if self.stock_price_library is None:
            #pre-processing
            orders_df = orders_df.sort_values(['Date'])
            orders_column = orders_df['Date']
            start_date = orders_column.head(1).dt
            end_date = orders_column.tail(1).dt
            start_date = dt.datetime(start_date.year, start_date.month, start_date.day)
            end_date = dt.datetime(end_date.year, end_date.month, end_date.day)

            list_of_symbols = orders_df.Symbol.unique().tolist()  # This really makes me wish Python was typesafe

            self.stock_price_library = market.get_stock_prices(list_of_symbols, start_date, end_date)

        portvals_raw = market.compute_portvals_abridged_faster(orders_df, commission=self.commission,
                                                               impact=self.impact, stock_price_library=self.stock_price_library)

        # clean up
        portvals = market.extract_portvals_only(portvals_raw)

        cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = market.compute_portval_stats(portvals, rfr=0.0, sf=252, sv=self.sv)
        if sharpe_ratio <=0:
            return sharpe_ratio*-10e6 #some big number we don't want
        print("sharpe is ", sharpe_ratio, "for ", self.symbol)
        return float(1)/sharpe_ratio #minimizing the inverse of sharpe will maximize sharpe

    def optimize_me(self):
        #run optimization algorithms

        #variables = copy.deepcopy(self.variables)
        #res = spo.minimize(self.minimize_this, variables, method='SLSQP', options={'disp':True})
        #minimizer_kwargs = {"method": "BFGS"}
        #bnds = ((0, 2), (0, 2), (0, 40), (50, 100), (0, 1))
        # #minimizer_kwargs = {"method": "SLSQP", "bounds":bnds}
        #minimizer_kwargs = {"method":"L-BFGS-B", "bounds":bnds}
        #res = spo.basinhopping(self.minimize_this, variables, minimizer_kwargs=minimizer_kwargs,
                             #  T=10e6, stepsize=10, disp=True, interval=5, niter=50)

        bnds = ((0, 5), (0, 5), (0, 100), (0, 100), (0, 1))
        #minimizer_kwargs = {"method": "SLSQP", "bounds":bnds}
        minimizer_kwargs = {"method":"L-BFGS-B", "bounds":bnds}
        #minimizer_kwargs = {"method":"SLSQP", "bounds":bnds, "ftol":0.1, "eps": 1e-3, "maxiter":2}
        my_bounds = MyBounds()
        mytakestep = MyTakeStep()
        variables = self.variables #initial guess
        start_time = time.time()
        res = spo.basinhopping(self.minimize_this, variables,
                               T=10e12, disp=True, niter=1)
        print("total opt time is ", time.time()-start_time)
        #res = spo.minimize(self.minimize_this, variables, method='SLSQP', options={'disp': True})

        # rranges = (slice(0.5, 2, 0.25), slice(0.5, 2, 0.25), slice(0, 50, 10), slice(50, 100, 10), slice(0, 1, 0.1))
        # res_brute = spo.brute(self.minimize_this, rranges, full_output=True, finish=spo.fmin)
        #self.variables = res.x
        self.variables = res.x

        print "optimal variables are ", self.variables

    def run_without_training(self, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=100000):
        orders_df = self.testPolicy(symbol=self.symbol, sd=sd, ed=ed, sv=sv, variables=self.variables)
        return orders_df

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31),
                   sv=100000, variables=[1, 1, 20, 80, 0.7]):
        print variables
        orders_df = market.make_df_to_match_trading_days(colnames=['Date','Symbol', 'Order', 'Shares'],
                                                         symbol='JPM', sd=sd, ed=ed)

        stock_price_library = market.get_stock_prices([symbol], sd, ed)

        #iterate through stock price library and short before all the drops and buy for all the gains

        orders_df['Symbol'] = symbol #assign to JPM
        orders_df['Order'] = "" #initialize with no orders
        orders_df['Shares'] = 0.0 #initialize with no shares
        orders_df['Date'] = orders_df.index

        state = 0
        short_price = 0.0
        days_in_short = 0

        #load indicators
        bollinger_df = indicators.get_bollinger(symbol, sd=sd, ed=ed)
        STOK_df = indicators.stochastic_oscillator_20(symbol, sd=sd, ed=ed)
        SMA_df = indicators.SMA(symbol, sd=sd, ed=ed)

        for row in orders_df.itertuples(index=True):
            order_date = getattr(row, 'Date')
            symbol = getattr(row, 'Symbol')
            # print "The symbol is ", symbol
            order = getattr(row, 'Order')
            shares = getattr(row, 'Shares')

            bollinger_high_today = bollinger_df.loc[order_date]['bb1']
            bollinger_low_today = bollinger_df.loc[order_date]['bb2']
            STOK_D_today = STOK_df.loc[order_date]['%D']
            stock_price_today = stock_price_library.loc[order_date][symbol]
            P_over_SMA_today = SMA_df.loc[order_date]['P/SMA']

            total_votes = []
            bb_vote = 0
            bb_weight = 1
            stochastic_vote = 0
            stochastic_weight = 1
            sma_vote = 0
            sma_weight = 1

            #load in the variables
            if stock_price_today > bollinger_high_today*variables[0]:
                bb_vote = -1 #sell!

            elif stock_price_today < bollinger_low_today*variables[1]:
                bb_vote = 1 #buy!

            if STOK_D_today < variables[2]:
                stochastic_vote = 1 #buy!

            elif STOK_D_today > variables[3]:
                stochastic_vote = -1 #get out of there!

            if P_over_SMA_today < variables[4]:
                 sma_vote = -1 #looks like a selling opp


            total_votes.append(bb_vote*bb_weight)
            total_votes.append(stochastic_vote*stochastic_weight)
            #total_votes.append(sma_vote*sma_weight)

            average_vote = sum(total_votes)/len(total_votes)

            if state < 0:
                days_in_short += 1

            if state < 0 and stock_price_today > 1.2*short_price or days_in_short > 45:
                #if the price has risen 20% above our short price
                #or if we have been in a short position for more than 45
                #get out of our position and wait for next move.
                orders_df.set_value(order_date, 'Order', 'BUY')
                orders_df.set_value(order_date, 'Shares', abs(0 - state))
                state = 0 #reset
                days_in_short = 0

            else:

                if average_vote < 0: #this means I want to sell
                    if abs(-1000 - state) == 0: #legality check for sale transaction
                        pass
                    else:
                        orders_df.set_value(order_date, 'Order', 'SELL')
                        orders_df.set_value(order_date, 'Shares', abs(-1000 - state))
                        short_price = stock_price_today
                        state = -1000 #short state

                if average_vote > 0: #this means I want to buy
                    if abs(1000 - state) == 0: #legality check for buy transaction
                        pass
                    else:
                        orders_df.set_value(order_date, 'Order', 'BUY')
                        orders_df.set_value(order_date, 'Shares', abs(1000 - state))
                        state = 1000 #long state
                        days_in_short = 0

                else:
                    pass
            #my best policy (based on seeing the future)

        return orders_df


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"

