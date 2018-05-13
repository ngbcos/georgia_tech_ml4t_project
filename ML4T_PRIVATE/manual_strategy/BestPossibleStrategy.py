import datetime as dt
import marketsimcode as market

def testPolicy(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
    orders_df = market.make_df_to_match_trading_days(colnames=['Date','Symbol', 'Order', 'Shares'],
                                                     symbol='JPM', sd=sd, ed=ed)

    stock_price_library = market.get_stock_prices([symbol], sd, ed)
    day_to_day_difference = stock_price_library.diff()
    day_to_day_difference = day_to_day_difference[symbol].shift(-1) #negative values mean the price will decrease tomorrow

    #iterate through stock price library and short before all the drops and buy for all the gains

    orders_df['Symbol'] = symbol #assign to JPM
    orders_df['Order'] = "" #initialize with no orders
    orders_df['Shares'] = 0.0 #initialize with no shares
    orders_df['Date'] = orders_df.index


    state = 0

    for row in orders_df.itertuples(index=True):
        order_date = getattr(row, 'Date')
        symbol = getattr(row, 'Symbol')
        # print "The symbol is ", symbol
        order = getattr(row, 'Order')
        shares = getattr(row, 'Shares')

        if day_to_day_difference.loc[order_date] < 0: #this means the next day will show a decline
            orders_df.set_value(order_date, 'Order', 'SELL')
            orders_df.set_value(order_date, 'Shares', abs(-1000 - state))
            state = -1000 #short state

        if day_to_day_difference.loc[order_date] > 0: #this means the next day will show a rise
            orders_df.set_value(order_date, 'Order', 'BUY')
            orders_df.set_value(order_date, 'Shares', abs(1000 - state))
            state = 1000 #long state
    #my best policy (based on seeing the future)

    return orders_df




