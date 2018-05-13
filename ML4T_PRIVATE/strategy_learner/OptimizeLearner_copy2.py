"""
Template for implementing Optimize Learner
"""

import numpy as np
import random
import copy
import datetime as dt
import scipy.optimize as spo
import pandas as pd
import util

import marketsimcode as market
import indicators as indicators
import time

class RandomDisplacementBounds(object):
    """random displacement with bounds"""
    def __init__(self, xmin, xmax, stepsize=0.5):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds"""
        while True:
            # this could be done in a much more clever way, but it will work for example purposes
            xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
            if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
                break
        return xnew



class OptimizeLearner(object):
    def __init__(self, \
                 variables = [0.5, 0.2, 1], \
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
        self.cash = sv
        self.sd = sd
        self.ed = ed
        self.verbose = verbose
        self.commission = commission
        self.impact = impact

        self.shares = 0

        self.stock_price_library = None #attempt to pre-cache something


        dates = pd.date_range(sd - dt.timedelta(40), ed)
        df = util.get_data([symbol], dates)[symbol]
        #print df.index
        normalized = df #/df.ix[sd:][0]#df.loc[sd]#df.ix[0]
        sma = normalized.rolling(20).mean()
        std = normalized.rolling(20).std()
        bb = (normalized - (sma - std * 2)) / ((sma + std * 2) - (sma - std * 2))

        #this is very hacky
        STOK_df = indicators.stochastic_oscillator_20(symbol, sd=sd, ed=ed)
        STOKD = STOK_df['%D']

        df = pd.DataFrame(df).assign(normalized=normalized).assign(SMA=sma).assign(PSMA=normalized / sma).assign(STD20=std).assign(STOKD=STOKD)[sd:]
        daily_returns = df[symbol].copy()
        daily_returns[1:] = (df[symbol].ix[1:] / df[symbol].ix[:-1].values) - 1  # from lecture
        daily_returns.ix[0] = 0
        df = df.assign(dr=daily_returns)

        self.market = df
        #print df
        self.current_stats = self.market
        #print(self.current_stats)

        self.trades = pd.DataFrame(index=df.index)
        self.trades['trade'] = 0.0

    def buy(self, trading_day):
        if self.shares >= 1000:#abs(1000 - self.shares) == 0:  # legality check for buy transaction
            pass
        else:
            closing_price = self.current_stats[self.symbol]
            # self.price = pd.cut(self.df['normalized'], 10, labels=False)[self.current_stats[0]]
            self.price = closing_price #pd.cut(self.df[self.symbol], 10, labels=False)[self.current_stats[0]]
            buy_amount = 1000
            if self.verbose: print "Buy %f @ %0.2f" % (buy_amount, closing_price)
            self.shares = self.shares + buy_amount
            self.cash = self.cash - (buy_amount * closing_price*(1+self.impact))  # Closing price
            dr = self.shares * self.current_stats['dr']

            self.trades.loc[trading_day]['trade'] = buy_amount
            #print "bought, now I have ", self.shares

        #return dr  # Reward


    def sell(self, trading_day):
        if self.shares <= -1000:  # legality check for sale transaction
            pass
        else:
            sell_amount = 1000#1000 - self.shares
            self.shares = self.shares - sell_amount
            if self.verbose: print "SELL %f @ %0.2f" % (sell_amount, self.current_stats[self.symbol])
            self.cash = self.cash + (sell_amount * self.current_stats[self.symbol]*(1-self.impact))  # Closing price
            self.price = 0  # Reset Purchase Price

            self.trades.loc[trading_day]['trade'] = -sell_amount


    def cash_out(self, trading_day):
        if self.shares < 0:
            closing_price = self.current_stats[self.symbol]
            self.price = closing_price  # pd.cut(self.df[self.symbol], 10, labels=False)[self.current_stats[0]]
            buy_amount = abs(self.shares)
            if self.verbose: print "Buy %f @ %0.2f" % (buy_amount, closing_price)
            self.shares = self.shares + buy_amount
            self.cash = self.cash - (buy_amount * closing_price * (1 + self.impact))  # Closing price
            dr = self.shares * self.current_stats['dr']
            self.trades.loc[trading_day]['trade'] = buy_amount


    def minimize_this(self, variables):
        #print variables
        orders_df = self.testPolicy(symbol=self.symbol, sd=self.sd, ed=self.ed, sv=self.sv, variables=variables)

        #attempting to make this faster
        if self.stock_price_library is None:
            self.stock_price_library = market.get_stock_prices([self.symbol], self.sd, self.ed)

        portvals_raw = market.compute_portvals_abridged_faster(orders_df, commission=self.commission,
                                                               impact=self.impact, stock_price_library=self.stock_price_library)

        # sddr = daily_returns.std()
        #
        # # back calculate daily risk free rate given the annual risk free rate
        # daily_rfr = (1 + rfr) ** (1 / float(252)) - 1
        #
        # sr = (daily_returns - daily_rfr).mean() / (daily_returns - daily_rfr).std()
        #
        # # Convert to annualized sharpe ratio
        # K = sf ** (1 / float(2))
        # sr = K * sr

        # clean up
        portvals = market.extract_portvals_only(portvals_raw)

        cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = market.compute_portval_stats(portvals, rfr=0.0, sf=252, sv=self.sv)
        if sharpe_ratio <=0:
            return sharpe_ratio*-10e6 #some big number we don't want
        print("sharpe is ", sharpe_ratio, "for ", self.symbol)
        print("cumulative return is ", cum_ret)
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

        # rewrite the bounds in the way required by L-BFGS-B
        xmin=[0, 0, 0]
        xmax = [1, 1, 1]
        bounds = [(low, high) for low, high in zip(xmin, xmax)]
        minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)

        # define the new step taking routine and pass it to basinhopping
        take_step = RandomDisplacementBounds(0, 1)
        result = spo.basinhopping(self.minimize_this, self.variables, niter=1, minimizer_kwargs=minimizer_kwargs,
                              take_step=take_step)
        print result

        #minimizer_kwargs = {"method": "SLSQP", "bounds":bounds}
        #minimizer_kwargs = {"method":"L-BFGS-B", "bounds":bnds}
        #minimizer_kwargs = {"method":"SLSQP", "bounds":bnds, "ftol":0.1, "eps": 1e-3, "maxiter":2}
        # my_bounds = MyBounds()
        # mytakestep = MyTakeStep()
        # variables = self.variables #initial guess
        # #start_time = time.time()
        # print('birds')
        # res = spo.basinhopping(self.minimize_this, variables,
        #                         T=100, disp=True, niter=1)

        # print("total opt time is ", time.time()-start_time)
        #res = spo.minimize(self.minimize_this, variables, method='SLSQP', options={'disp': True})

        # rranges = (slice(0.5, 1.5, 0.5), slice(0, 31, 10), slice(0.7, 1, 0.1))
        # res_brute = spo.brute(self.minimize_this, rranges, full_output=True, finish=spo.fmin)
        # self.variables = res_brute[0]
        self.variables = result.x

        print "optimal variables are ", self.variables

    # def reset_trades(self, dates):
    #     print dates
    #     df = util.get_data([self.symbol], dates)[self.symbol]
    #     trades = pd.DataFrame(index=df.index)
    #     trades['trade'] = 0.0
    #     #print("returning empty trades df")
    #     return trades

    # def convert_df_to_trades(self, orders_df):
    #     trades = pd.DataFrame(index=orders_df.index)
    #     trades['trades'] = orders_df[]

    def recalculate_metrics(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1)):
        dates = pd.date_range(sd - dt.timedelta(40), ed)
        df = util.get_data([symbol], dates)[symbol]
        #print df.index
        normalized = df #/df.ix[sd:][0]#df.loc[sd]#df.ix[0]
        sma = normalized.rolling(20).mean()
        std = normalized.rolling(20).std()
        bb = (normalized - (sma - std * 2)) / ((sma + std * 2) - (sma - std * 2))

        #this is very hacky
        STOK_df = indicators.stochastic_oscillator_20(symbol, sd=sd, ed=ed)
        STOKD = STOK_df['%D']

        df = pd.DataFrame(df).assign(normalized=normalized).assign(SMA=sma).assign(PSMA=normalized / sma).assign(STD20=std).assign(STOKD=STOKD)[sd:]
        daily_returns = df[symbol].copy()
        daily_returns[1:] = (df[symbol].ix[1:] / df[symbol].ix[:-1].values) - 1  # from lecture
        daily_returns.ix[0] = 0
        df = df.assign(dr=daily_returns)

        self.market = df
        self.current_stats = self.market

        self.trades = pd.DataFrame(index=df.index)
        self.trades['trade'] = 0.0 #reset trades

        #print("Reset statistics")


    def run_without_training(self, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=100000):
        dates = pd.date_range(sd, ed)
        # self.trades = self.reset_trades(dates=dates)
        # print "reset trades"

        self.recalculate_metrics(self.symbol, sd=sd, ed=ed)
        orders_df = self.testPolicy(symbol=self.symbol, sd=sd, ed=ed, sv=sv, variables=self.variables)

        #super hacky
        #print orders_df
        return self.trades#orders_df

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31),
                   sv=100000, variables=[1, 20]):
        print variables
        orders_df = market.make_df_to_match_trading_days(colnames=['Date', 'Symbol', 'Order', 'Shares'],
                                                           symbol=self.symbol, sd=sd, ed=ed)

        print("Test Policy refreshed orders DF")


        orders_df['Symbol'] = symbol #assign to JPM
        orders_df['Order'] = "" #initialize with no orders
        orders_df['Shares'] = 0.0 #initialize with no shares
        orders_df['Date'] = orders_df.index

        short_price = 0.0
        days_in_short = 0

        # bollinger_df['bb1'] = SMA_df['SMA'] + (STD_rolling['STD20'] * 2)
        # bollinger_df['bb2'] = SMA_df['SMA'] - (STD_rolling['STD20'] * 2)

        #load indicators
        #bollinger_df = indicators.get_bollinger(symbol, sd=sd, ed=ed)
        #STOK_df = indicators.stochastic_oscillator_20(symbol, sd=sd, ed=ed)
        #SMA_df = indicators.SMA(symbol, sd=sd, ed=ed)

        for row in orders_df.itertuples(index=True):
            order_date = getattr(row, 'Date')
            symbol = getattr(row, 'Symbol')
            # print "The symbol is ", symbol
            order = getattr(row, 'Order')
            shares = getattr(row, 'Shares')

            self.current_stats = self.market.loc[order_date]

            bollinger_high_today = self.current_stats['SMA'] + (self.current_stats['STD20'] * 2)*variables[0]*2
            bollinger_low_today = self.current_stats['SMA'] - (self.current_stats['STD20'] * 2)*variables[0]*2
            STOK_D_today = self.current_stats['STOKD']#STOK_df.loc[order_date]['%D']
            stock_price_today = self.current_stats[symbol]
            P_over_SMA_today = self.current_stats['PSMA']

            total_votes = []
            bb_vote = 0
            bb_weight = 1
            stochastic_vote = 0
            stochastic_weight = 1
            sma_vote = 0
            sma_weight = 0.5

            #load in the variables
            if stock_price_today > bollinger_high_today:
                bb_vote = -1 #sell!

            elif stock_price_today < bollinger_low_today:
                bb_vote = 1 #buy!

            if STOK_D_today < variables[1]*100:
                stochastic_vote = 1 #buy!

            elif STOK_D_today > (100-variables[1]*100):
                stochastic_vote = -1 #get out of there!

            if P_over_SMA_today < variables[2]:
                  sma_vote = -1 #looks like a selling opp

            total_votes.append(bb_vote*bb_weight)
            #total_votes.append(stochastic_vote*stochastic_weight)
            total_votes.append(sma_vote*sma_weight)

            average_vote = sum(total_votes)/len(total_votes)

            if self.shares < 0:
                days_in_short += 1

            if self.shares < 0 and stock_price_today > 1.2*short_price or days_in_short > 45:
                #if the price has risen 20% above our short price
                #or if we have been in a short position for more than 45
                #get out of our position and wait for next move.
                orders_df.set_value(order_date, 'Order', 'BUY')
                orders_df.set_value(order_date, 'Shares', abs(self.shares))
                self.cash_out(order_date)
                #self.shares = 0 #reset
                days_in_short = 0

            else:

                if average_vote < 0: #this means I want to sell
                    if abs(-1000 - self.shares) == 0: #legality check for sale transaction
                        pass
                    else:
                        self.sell(trading_day=order_date)
                        orders_df.set_value(order_date, 'Order', 'SELL')
                        orders_df.set_value(order_date, 'Shares', 1000)
                        short_price = stock_price_today
                        #self.shares = -1000 #short self.shares

                if average_vote > 0: #this means I want to buy
                    if abs(1000 - self.shares) == 0: #legality check for buy transaction
                        pass
                    else:
                        self.buy(trading_day=order_date)
                        orders_df.set_value(order_date, 'Order', 'BUY')
                        orders_df.set_value(order_date, 'Shares', 1000)
                        #self.shares = 1000 #long self.shares
                        days_in_short = 0

                else:
                    pass

            if abs(self.shares) > 1000:
                print "This should never happen"


            #my best policy (based on seeing the future)
                #print orders_df
        #print orders_df.index
        #print orders_df
        self.trades.to_csv('test.csv')
        return orders_df


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"

