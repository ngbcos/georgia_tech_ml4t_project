# Nick Lee - nlee68

import matplotlib
matplotlib.use('Agg') #allows code to run on Buffett Machines without crashing Linux

import datetime as dt
import marketsimcode as market
import ManualStrategy as ms
import indicators as ind
import StrategyLearner as sl

import matplotlib.pyplot as plt


def author():
    return 'nlee68'  # replace tb34 with your Georgia Tech username

def plot_code(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31)):
    startval = 100000
    # orders_df = bps.testPolicy(sd=start_date, ed=end_date)
    plt.figure()  # I'm going to try to cheat

    orders_df = ms.testPolicy(sd=sd, ed=ed, sv=startval)
    benchmark_policy = ind.benchmark_policy(sd=sd, ed=ed, sv=startval)

    #get strat learner orders
    strat_learner = sl.StrategyLearner(verbose=False, impact=0)
    strat_learner.addEvidence(symbol="JPM", sd=sd, ed=ed, sv=startval)
    strat_orders_df = strat_learner.testPolicy(sd=sd, ed=ed, sv=startval)
    strat_orders_df = strat_learner.get_classic_policy_for_testing()


    benchmark_policy.to_csv('benchmark.csv')
    orders_df.to_csv('orders.csv')
    strat_orders_df.to_csv('strat_orders.csv')

    # portvals_raw = compute_portvals_abridged(orders_df, commission=0, impact=0)
    portvals_raw = market.compute_portvals_abridged(orders_df, commission=0, impact=0)
    benchmark_portvals_raw = market.compute_portvals_abridged(benchmark_policy, commission=0, impact=0.0)
    strat_portvals_raw = market.compute_portvals_abridged(strat_orders_df, commission=0, impact=0.0)

    # clean up
    portvals = market.extract_portvals_only(portvals_raw)
    benchmark_portvals = market.extract_portvals_only(benchmark_portvals_raw)
    strat_portvals = market.extract_portvals_only(strat_portvals_raw)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = market.compute_portval_stats(portvals, rfr=0.0, sf=252, sv=startval)
    bench_cum_ret, bench_avg_daily_ret, bench_std_daily_ret, bench_sharpe_ratio = market.compute_portval_stats(
        benchmark_portvals, rfr=0.0, sf=252, sv=startval)
    learn_cum_ret, learn_avg_daily_ret, learn_std_daily_ret, learn_sharpe_ratio = market.compute_portval_stats(
        strat_portvals, rfr=0.0, sf=252, sv=startval)

    # # Get SPY data
    # symbols = ['SPY']
    # allocations = [1]
    # start_val = startval0
    # risk_free_rate = 0.0
    # sample_freq = 252
    #
    # # Assess the portfolio of SPY
    # cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY, ev = market.assess_portfolio(sd=sd,
    #                                                                                            ed=ed,
    #                                                                                            syms=symbols,
    #                                                                                            allocs=allocations,
    #                                                                                            sv=startval,
    #                                                                                            gen_plot=False)

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(sd, ed)
    print
    print "Sharpe Ratio of Manual Strategy: {}".format(sharpe_ratio)
    print "Sharpe Ratio of Benchmark: {}".format(bench_sharpe_ratio)
    print "Sharpe Ratio of Strategy Learner: {}".format(learn_sharpe_ratio)
    print
    print "Cumulative Return of Manual Strategy: {}".format(cum_ret)
    print "Cumulative Return of Benchmark: {}".format(bench_cum_ret)
    print "Cumulative Return of Strategy Learner : {}".format(learn_cum_ret)
    print
    print "Standard Deviation of Manual Strategy: {}".format(std_daily_ret)
    print "Standard Deviation of Benchmark: {}".format(bench_std_daily_ret)
    print "Standard Deviation of Strategy Learner : {}".format(learn_std_daily_ret)
    print
    print "Average Daily Return of Manual Strategy: {}".format(avg_daily_ret)
    print "Average Daily Return of Benchmark: {}".format(bench_avg_daily_ret)
    print "Average Daily Return of Strategy Learner : {}".format(learn_avg_daily_ret)
    print
    print "Final Manual Strategy Value: {}".format(portvals[-1])
    print "Final Benchmark Value: {}".format(benchmark_portvals[-1])
    print "Final Strategy Learner Value: {}".format(strat_portvals[-1])

    benchmark_normalized = market.normalize_data(benchmark_portvals_raw)
    benchmark_normalized = market.extract_portvals_only(benchmark_normalized)
    best_portfolio_normalized = market.normalize_data(portvals_raw)
    best_portfolio_normalized = market.extract_portvals_only(best_portfolio_normalized)
    strat_portfolio_normalized = market.normalize_data(strat_portvals_raw)
    strat_portfolio_normalized = market.extract_portvals_only(strat_portfolio_normalized)

    stock_library = market.make_df_to_match_trading_days(colnames=['Date', 'Value'], symbol='JPM',
                                                  sd=sd, ed=ed)

    benchmark_line = plt.plot(stock_library.index, benchmark_normalized.values, label="Benchmark")
    plt.setp(benchmark_line, linestyle='-', color='b', linewidth=1.0)
    best_portfolio_line = plt.plot(stock_library.index, best_portfolio_normalized.values, label="Manual Strategy")
    plt.setp(best_portfolio_line, linestyle='-', color='k', linewidth=3.0)
    strat_portfolio_line = plt.plot(stock_library.index, strat_portfolio_normalized.values, label="Learner Strategy")
    plt.setp(strat_portfolio_line, linestyle='-', color='r', linewidth=1.0)
    legend = plt.legend(loc='best', shadow=True)
    plt.title("Normalized chart for JPM, in-sample period")
    plt.ylabel("Normalized Value")
    plt.grid()
    plt.savefig("experiment1.png")


if __name__ == "__main__":

    print("Working... Please be patient. Plots will be saved to disk.")
    plot_code(sd=dt.datetime(2018,1,1), ed=dt.datetime(2018,03,12))

    # #show bollinger and SMA on stock price
    # fig = plt.figure()
    #
    # #in sample
    # start_date = dt.datetime(2008,1,1)
    # end_date = dt.datetime(2009,12,31)
    # symbol = "JPM"
    #
    # plot_prices(sd=start_date, ed=end_date)
    # plot_SMA(sd=start_date, ed=end_date)
    # plot_bollinger(sd=start_date, ed=end_date)
    # legend = plt.legend(loc='best', shadow=True)
    #
    # plt.grid()
    # plt.savefig("final.png")
    # plt.show()
    # #
    # # fig2 = plt.figure(2)
    # # plot_STOK()
    # # plt.legend(loc='best', shadow=True)
    # # plt.show()
    #
    # #show STOKD and JP Morgan subplots
    # plt.figure(1)
    # plt.subplot(211)
    # plot_STOK(sd=start_date, ed=end_date)
    # plt.title("Stochastic Oscillator vs Time for JPY")
    # plt.legend(loc='best', shadow=True)
    #
    # plt.subplot(212)
    # plot_prices()
    # plt.show()

