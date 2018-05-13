# Nick Lee - nlee68

import datetime as dt
import scipy.optimize as spo
import pandas as pd
import util

import marketsimcode as market
import indicators as indicators


def author():
    return 'nlee68'  # replace tb34 with your Georgia Tech username

class OptimizeLearner(object):
    def __init__(self, \
                 variables = [1, 0.5, 1], \
                 symbol="JPM", \
                 sd=dt.datetime(2008, 1, 1), \
                 ed=dt.datetime(2009, 1, 1),
                 commission = 0,
                 impact = 0,
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

        self.recalculate_metrics(self.symbol, self.sd, self.ed)

        self.classic_orders_df = None  # for legacy marketsim code


    def buy(self, trading_day):
        if self.shares < 1000:  # legality check for buy transaction
            #buy
            closing_price = self.current_stats[self.symbol]
            self.price = closing_price  # pd.cut(self.df[self.symbol], 10, labels=False)[self.current_stats[0]]
            buy_amount = abs(1000 - self.shares)
            if self.verbose: print "Buy %f @ %0.2f" % (buy_amount, closing_price)
            self.shares = self.shares + buy_amount
            self.cash = self.cash - (buy_amount * closing_price * (1 + self.impact))  # Closing price
            self.trades.loc[trading_day]['trade'] = buy_amount
        else:
            pass #don't buy a thing


    def sell(self, trading_day):
        if self.shares > -1000:  # legality check for sale transaction
            sell_amount = abs(-1000 - self.shares)
            self.shares = self.shares - sell_amount
            if self.verbose: print "SELL %f @ %0.2f" % (sell_amount, self.current_stats[self.symbol])
            self.cash = self.cash + (sell_amount * self.current_stats[self.symbol] * (1 - self.impact))  # Closing price
            self.price = 0  # Reset Purchase Price

            self.trades.loc[trading_day]['trade'] = -sell_amount
        else:
            pass


    def cash_out(self, trading_day):
        if self.shares < 0:
            closing_price = self.current_stats[self.symbol]
            self.price = closing_price
            buy_amount = abs(self.shares)
            if self.verbose: print "Buy %f @ %0.2f" % (buy_amount, closing_price)
            self.shares = self.shares + buy_amount
            self.cash = self.cash - (buy_amount * closing_price * (1 + self.impact))  # Closing price
            self.trades.loc[trading_day]['trade'] = buy_amount


    def minimize_this(self, variables):
        if self.verbose:
            print variables

        orders_df = self.testPolicy(symbol=self.symbol, sd=self.sd, ed=self.ed, sv=self.sv, variables=variables)

        #attempting to make this faster
        if self.stock_price_library is None:
            self.stock_price_library = market.get_stock_prices([self.symbol], self.sd, self.ed)

        portvals_raw = market.compute_portvals_abridged_faster(orders_df, commission=self.commission,
                                                               impact=self.impact, stock_price_library=self.stock_price_library)

        portvals = portvals_raw[portvals_raw.columns[0]]

        cum_ret = portvals[-1]/float(self.sv) - 1 #lightning optimization

        return -cum_ret #I apply a negative so negative returns are positive, and positive returns are negative


    def get_classic_orders(self):
        return self.classic_orders_df

    def optimize_me(self):

        #rranges = (slice(0.5, 1.3, 0.25), slice(0.2, 0.45, 0.2), slice(0.7, 1.15, 0.15))
        rranges = (slice(0.75, 1.3, 0.25), slice(0, 1.1, 1), slice(0.7, 1.01, 0.15), slice(0,0.51,0.5))
        #rranges = (slice(0.75, 2.1, 0.25), slice(0, 1.1, 1), slice(0.7, 1.25, 0.15), slice(0,1.1,0.5))

        res_brute = spo.brute(self.minimize_this, rranges, full_output=True, finish=None)
        # res_brute = spo.brute(self.minimize_this, rranges,
        #       finish=lambda func, x0, args=(): spo.fmin(func, x0, args, full_output=True, maxiter=1))

        self.variables = res_brute[0]
        if self.verbose:
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
        prices_df = df #/df.ix[sd:][0]#df.loc[sd]#df.ix[0]
        sma = prices_df.rolling(20).mean()
        std = prices_df.rolling(20).std()
        bb = (prices_df - (sma - std * 2)) / ((sma + std * 2) - (sma - std * 2))

        #this is very hacky
        STOK_df = indicators.stochastic_oscillator_20(symbol, sd=sd, ed=ed)
        STOKD = STOK_df['%D']

        df = pd.DataFrame(df).assign(prices=prices_df).assign(SMA=sma).assign(PSMA=prices_df / sma).assign(STD20=std).assign(STOKD=STOKD)[sd:]
        daily_returns = df[symbol].copy()
        daily_returns[1:] = (df[symbol].ix[1:] / df[symbol].ix[:-1].values) - 1  # from lecture
        daily_returns.ix[0] = 0
        df = df.assign(dr=daily_returns)

        self.market = df
        self.current_stats = self.market

        self.trades = pd.DataFrame(index=df.index)
        self.trades['trade'] = 0.0 #reset trades

        self.shares = 0 #reset the sharecount for self... I think this fixes my bugs.

        if self.verbose:
            print("Reset statistics")


    def run_without_training(self, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=100000):
        if self.verbose:
            print("Running without training. My variables are ", self.variables)

        self.recalculate_metrics(self.symbol, sd=sd, ed=ed)
        orders_df = self.testPolicy(symbol=self.symbol, sd=sd, ed=ed, sv=sv, variables=self.variables)

        self.classic_orders_df = orders_df #for legacy marketsim code
        return self.trades#orders_df

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31),
                   sv=100000, variables=[1, 1, 0.7, 0.5]):
        #print variables
        orders_df = market.make_df_to_match_trading_days(colnames=['Date', 'Symbol', 'Order', 'Shares'],
                                                           symbol=self.symbol, sd=sd, ed=ed)

        #print("Test Policy refreshed orders DF")


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
            # order = getattr(row, 'Order')
            # shares = getattr(row, 'Shares')

            self.current_stats = self.market.loc[order_date]

            bollinger_high_today = self.current_stats['SMA'] + (self.current_stats['STD20'] * 2)*variables[0]
            bollinger_low_today = self.current_stats['SMA'] - (self.current_stats['STD20'] * 2)*variables[0]
            STOK_D_today = self.current_stats['STOKD']#STOK_df.loc[order_date]['%D']
            stock_price_today = self.current_stats[symbol]
            P_over_SMA_today = self.current_stats['PSMA']

            total_votes = []
            bb_vote = 0
            bb_weight = 1
            stochastic_vote = 0
            stochastic_weight = variables[1]
            sma_vote = 0
            sma_weight = variables[3]

            #load in the variables
            if stock_price_today > bollinger_high_today:
                bb_vote = -1 #sell!

            elif stock_price_today < bollinger_low_today:
                bb_vote = 1 #buy!

            if STOK_D_today < 20:
                stochastic_vote = 1 #buy!

            elif STOK_D_today > 80:
                stochastic_vote = -1 #get out of there!

            if P_over_SMA_today < variables[2]:
                  sma_vote = -1 #looks like a selling opp

            total_votes.append(bb_vote*bb_weight)
            if stochastic_weight > 0:
                total_votes.append(stochastic_vote*stochastic_weight)
            if sma_weight > 0:
                total_votes.append(sma_vote*sma_weight)

            average_vote = sum(total_votes)/len(total_votes)

            if self.shares < 0:
                days_in_short += 1

            if self.shares < 0 and stock_price_today > 1.2*short_price or days_in_short > 45:
                #if the price has risen 20% above our short price
                #or if we have been in a short position for more than 45
                #get out of our position and wait for next move.
                self.cash_out(order_date)
                orders_df.set_value(order_date, 'Order', 'BUY')
                orders_df.set_value(order_date, 'Shares', abs(self.trades.loc[order_date]['trade']))
                days_in_short = 0

            else:

                if average_vote < 0: #this means I want to sell
                    if abs(-1000 - self.shares) == 0: #legality check for sale transaction
                        pass
                    else:
                        self.sell(trading_day=order_date)
                        orders_df.set_value(order_date, 'Order', 'SELL')
                        orders_df.set_value(order_date, 'Shares', abs(self.trades.loc[order_date]['trade']))
                        short_price = stock_price_today
                        #self.shares = -1000 #short self.shares

                if average_vote > 0: #this means I want to buy
                    if abs(1000 - self.shares) == 0: #legality check for buy transaction
                        pass
                    else:
                        self.buy(trading_day=order_date)
                        orders_df.set_value(order_date, 'Order', 'BUY')
                        orders_df.set_value(order_date, 'Shares', abs(self.trades.loc[order_date]['trade']))
                        #self.shares = 1000 #long self.shares
                        days_in_short = 0

                else:
                    pass


            #my best policy (based on seeing the future)
                #print orders_df
        #print orders_df.index
        self.trades.to_csv('tradesTEST.csv')
        orders_df.to_csv('ordersTEST.csv')
        #self.classic_orders_df = orders_df  # for legacy marketsim code
        return orders_df


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
