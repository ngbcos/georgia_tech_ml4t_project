# Nick Lee - nlee68

import datetime as dt
import marketsimcode as market
import indicators as indicators

def author():
    return 'nlee68'  # replace tb34 with your Georgia Tech username

def testPolicy(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
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
        sma_weight = 0.5

        if stock_price_today > bollinger_high_today:
            bb_vote = -1 #sell!

        elif stock_price_today < bollinger_low_today:
            bb_vote = 1 #buy!

        if STOK_D_today < 20:
            stochastic_vote = 1 #buy!

        elif STOK_D_today > 80:
            stochastic_vote = -1 #get out of there!

        if P_over_SMA_today < 0.7:
             sma_vote = -1 #looks like a selling opp


        total_votes.append(bb_vote*bb_weight)
        total_votes.append(stochastic_vote*stochastic_weight)
        total_votes.append(sma_vote*sma_weight)

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
