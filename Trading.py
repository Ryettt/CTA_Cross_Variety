import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


class trading_strategy:
    def __init__(self, fcode1, fcode2, ic_train, weight_indicator, n, all_spread_p_return_daily, test_begin_date,
                 long_threshold = 0.8,
                 short_threshold = 0.2):
        # ic_train:ic for train_data
        # long_threshold: buy
        # short_threshold: sell
        # n: adjustment cycle
        # all_spread_p_return_daily:spread portfolio daily return of all data
        # test_begin_date: begining date of the test_period

        self.fcode1 = fcode1
        self.fcode2 = fcode2
        self.ic_train = ic_train
        self.weight_indicator = weight_indicator
        self.n = n
        self.all_return = all_spread_p_return_daily
        self.test_begin_date = test_begin_date
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def get_signal(self):
        # startdate for test
        percentage = ((self.weight_indicator).rolling(self.n * 2).apply(
            lambda x: x.rank(pct = True).iloc[-1])).iloc[
                     100:]  # drop first 100 days
        percentage.columns = ['percentage']
        self.signal = pd.DataFrame(data = np.zeros(len(percentage)), index = percentage.index, columns = ['signal'])

        if self.ic_train > 0:
            self.signal[(percentage >= self.long_threshold).values] = 1
            self.signal[(percentage <= self.short_threshold).values] = -1
        else:
            self.signal[(percentage >= self.long_threshold).values] = -1
            self.signal[(percentage <= self.short_threshold).values] = 1

        return self.signal

    def backtest(self, start_point = 0):
        # start_point:change the enter point
        # signal_to_trading:next day enter
        signal_n = (self.signal.shift(1)).dropna()
        self.signalf = (signal_n.iloc[start_point:]).copy(deep = True)
        signalp = self.signalf.copy(deep = True)  # use to index to get the final signal
        date_list = self.signalf.index.to_list()
        # get the first enter date:signal not 0
        enter_date = signalp[(signalp != 0).values].index[0]
        n_cycle = int(len(signalp.loc[enter_date:]) / self.n)
        while n_cycle >= 0:
            if n_cycle > 0:
                if len(signalp[(signalp != 0).values]) != 0:
                    enter_date = signalp[(signalp != 0).values].index[0]
                    enter_index = date_list.index(enter_date)
                    enter_index_for_signalp = (signalp.index.to_list()).index(enter_date)
                    # the next n days keep stable
                    self.signalf.iloc[enter_index:enter_index + self.n] = self.signalf.iloc[enter_index].values[0]
                    signalp = signalp.iloc[enter_index_for_signalp + self.n:]
                    # full self.n days after next enter day
                    n_cycle -= 1
                    continue
                else:
                    break
            if n_cycle == 0:
                if len(signalp[(signalp != 0).values]) != 0:
                    enter_date = signalp[(signalp != 0).values].index[0]
                    enter_index = date_list.index(enter_date)
                    # the next self.n days keep stable
                    self.signalf.iloc[enter_index:] = self.signalf.iloc[enter_index].values[0]
                    signalp = signalp.iloc[enter_index + self.n:]
                    n_cycle -= 1
                else:
                    break

        return self.signalf

    def get_nav(self, daily_return):
        nav = ((1 + daily_return).cumprod(axis = 0))
        return nav

    def get_compare_nav_all(self):
        # same index
        nav_t = self.get_nav(self.all_return.loc[self.signalf.index])
        nav_t.columns = ['nav_true']
        self.c_return_all = ((self.all_return.loc[self.signalf.index, self.all_return.columns[0]]).mul(
            self.signalf.iloc[:, 0])).to_frame()
        nav_c = self.get_nav(self.c_return_all)
        nav_c.columns = ['nav_construct']
        self.outcome = pd.concat([nav_t, nav_c], axis = 1)
        self.return_all = pd.concat([self.all_return.loc[self.signalf.index], self.c_return_all], axis = 1)
        self.return_all.columns = self.outcome.columns
        return self.outcome

    def get_compare_nav_test(self):
        # same index
        nav_t = self.get_nav(self.all_return.loc[self.test_begin_date:])
        nav_t.columns = ['nav_true']
        self.c_return_test = ((self.all_return.loc[self.test_begin_date:, self.all_return.columns[0]]).mul(
            self.signalf.loc[self.test_begin_date:, self.signalf.columns[0]])).to_frame()
        nav_c = self.get_nav(self.c_return_test)
        nav_c.columns = ['nav_construct']
        self.test_outcome = pd.concat([nav_t, nav_c], axis = 1, join = 'inner')
        self.return_test = pd.concat([self.all_return.loc[self.test_begin_date:], self.c_return_test], axis = 1,
                                     join = 'inner')
        self.return_test.columns = self.test_outcome.columns

        return self.test_outcome

    def get_max_drawdown(self):
        cummax_all = self.outcome.expanding().max()
        max_drawdown_all = ((cummax_all.sub(self.outcome).div(cummax_all)).max()).to_frame().T
        cummax_test = self.test_outcome.expanding().max()
        max_drawdown_test = (((cummax_test.sub(self.test_outcome)).div(cummax_test)).max()).to_frame().T
        max_drawdown = pd.concat([max_drawdown_all, max_drawdown_test], axis = 0)
        max_drawdown.index = ['max_drawdown_all', 'max_drawdown_test']
        return max_drawdown

    def get_shape_ratio(self):
        shape_ratio_all = (
                self.return_all.mean().div(self.return_all.std()) * np.sqrt(252)).to_frame().T
        shape_ratio_all.columns = self.return_all.columns
        shape_ratio_test = (self.return_test.mean().div(self.return_test.std()) * np.sqrt(
            252)).to_frame().T
        shape_ratio_test.columns = self.return_all.columns
        shape_ratio = pd.concat([shape_ratio_all, shape_ratio_test], axis = 0)
        shape_ratio.index = ['shape_ratio_all', 'shape_ratio_test']
        return shape_ratio

    def evaluation_outcome_plot(self):
        self.get_compare_nav_all()
        self.get_compare_nav_test()
        max_drawdown = self.get_max_drawdown()
        shape_ratio = self.get_shape_ratio()
        evaluation = pd.concat([max_drawdown, shape_ratio], axis = 0)

        self.outcome.plot()
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base = 100))
        plt.gcf().autofmt_xdate()
        plt.title("Nav_compare_all for " + self.fcode1 + " & " + self.fcode2)
        plt.savefig("./image/" + "Nav_compare_all for " + self.fcode1 + " & " + self.fcode2 + ", from " +
                    self.outcome.index.values[0] + ".png")
        plt.show()

        self.test_outcome.plot()
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base = 100))
        plt.gcf().autofmt_xdate()
        plt.title("Nav_compare_test for " + self.fcode1 + " & " + self.fcode2)
        plt.savefig("./image/" + "Nav_compare_test for " + self.fcode1 + " & " + self.fcode2 + ", from " +
                    self.outcome.index.values[0] + ".png")
        plt.show()

        return evaluation
