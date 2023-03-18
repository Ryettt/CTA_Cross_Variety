from get_future_pair import *


class Spread_portfolio:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.all_data = pd.concat([train_data, test_data])

    def get_ratio(self):
        model = lm.LinearRegression()
        model.fit(np.array(self.train_data.iloc[:, 0]).reshape(-1, 1),
                  np.array(self.train_data.iloc[:, 1]))
        self.ratio = model.coef_
        return self.ratio

    def get_spread_portfolio(self):
        # use the ratio get from train_data to get the spread_portfolio
        self.all_spread_portfolio = (self.all_data.iloc[:, 1] - self.get_ratio() * self.all_data.iloc[:, 0]).to_frame()
        self.all_spread_portfolio.columns = ["price"]

    def get_spead_p_return_f(self, n):
        # get future n days return
        self.all_spread_p_return_f = ((self.all_spread_portfolio.shift(
            -n) - self.all_spread_portfolio) / self.all_spread_portfolio).dropna()
        self.train_spread_p_return_f = (self.all_spread_p_return_f.loc[:self.train_data.index[-1], :]).dropna()
        self.test_spread_p_return_f = (self.all_spread_p_return_f.loc[self.test_data.index[0]:, :]).dropna()
        self.all_spread_p_return_f.columns = ["future " + str(n) + " days return"]
        self.train_spread_p_return_f.columns = ["future " + str(n) + " days return"]
        self.test_spread_p_return_f.columns = ["future " + str(n) + " days return"]

    def get_spead_p_return_true(self):
        # daily return of the spread portfolio
        self.all_spread_p_return_true = (self.all_spread_portfolio - self.all_spread_portfolio.shift(
            1)) / self.all_spread_portfolio.shift(1)
        self.all_spread_p_return_true.dropna(inplace = True)
        self.all_spread_p_return_true.columns = ['daily_return']
        return self.all_spread_p_return_true
