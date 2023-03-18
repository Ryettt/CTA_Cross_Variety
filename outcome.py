from Weight_indicator import *
from Get_spread_portfolio import *
from Trading import *

# only use close price
rawdata = pd.read_csv('./maincontact.csv', index_col = [2], parse_dates = True, usecols = [1, 3, 7])

# training data
# start_date = min(rawdata.loc[fcode1, 'TRADE_DT'][0], rawdata.loc[fcode2, 'TRADE_DT'][0])  # use earliest time
start_date = '2015-01-03'
end_date = '2020-12-31'


def get_all_outcome(fcode1, fcode2, nd, n):
    # nd:day lists to construct the indicators

    datapair = pair_future_data(rawdata, fcode1, fcode2, start_date, end_date)
    datapair.get_pair_data()
    all_data = datapair.all_data
    train_data = datapair.train_data
    test_data = datapair.test_data
    datapair.get_price_corr()
    datapair.plot_prices()
    datapair.plot_price_difference()
    datapair.contigration_test()

    index = pd.DataFrame()
    for i in nd:
        indicator = Indicator(all_data, i)
        index = pd.concat([index, indicator.get_all_indicators()], axis = 1, join = 'outer')
    index.dropna(inplace = True)

    # get the spread portfolio and return
    sp = Spread_portfolio(train_data, test_data)
    sp.get_spread_portfolio()

    sp.get_spead_p_return_f(n)
    fr_train = sp.train_spread_p_return_f
    fr_test = sp.test_spread_p_return_f
    all_spread_p_return_daily = sp.get_spead_p_return_true()
    train_merge_data = (pd.concat([fr_train, index], axis = 1, join = 'inner')).dropna()
    test_merge_data = (pd.concat([fr_test, index], axis = 1, join = 'inner')).dropna()

    # get weight and the weight indicator
    gw = Get_weight(fcode1, fcode2, train_merge_data, test_merge_data)
    w = gw.get_weight()
    gw.get_windicator_for_test_data()
    ind_e = gw.get_ind_efficiency()  # get the index_efficiency
    ind_e.to_csv("index_efficiency for " + fcode1 + " & " + fcode2 + ".csv")
    weight_indicator = gw.get_weight_indicator_for_all()
    # plot outcome
    gw.train_outcome_plot()
    gw.test_outcome_plot()

    # trading
    '''if ic_train>0:
    if weight_indicator>=long_threshold:long spread_portfolio the next day,
    and short it next day if weight_indicator<=short_threshold;
    if ic_train<0:
    then do the opposite
    '''
    ic_train = ind_e.loc['IC', 'train']
    long_threshold = 0.8
    short_threshold = 0.2
    test_start = test_data.index[0]
    ts = trading_strategy(fcode1, fcode2, ic_train, weight_indicator, n, all_spread_p_return_daily, test_start,
                          long_threshold, short_threshold)

    signal = ts.get_signal()
    # robust check
    evaluation = pd.DataFrame(columns = ['nav_true', 'nav_construct'])
    # change the enter point:
    for i in range(n):
        signalf = ts.backtest(i)
        outcome = ts.get_compare_nav_all()
        evaluation = pd.concat([evaluation, ts.evaluation_outcome_plot()], axis = 0)

    return evaluation


# wind code
fcode1 = 'Y.DCE'
fcode2 = 'M.DCE'
nd = [30, 60, 90, 120]
n = 22  # future n day(days) return and adjustment would equal to this
evaluation = get_all_outcome(fcode1, fcode2, nd, n)
evaluation.to_csv("./evaluation.csv")
