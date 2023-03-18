from get_future_pair import *
from pyfinance.ols import PandasRollingOLS

class Indicator:

    def __init__(self, data, n):
        # data: a dataframe containing the two futures close prices
        # n: past n days related to the indicator calculation
        self.data = data
        self.n = n  # n days index
        self.dataf1 = self.data.iloc[:, 0]
        self.dataf2 = self.data.iloc[:, 1]

    # indicator construction

    def calculate_ratio(self):  # calculate the ratio of price
        self.price_ratio = (self.dataf1 / self.dataf2).to_frame()  # calculate the ratio of prices

    def calculate_spread(self):  # calculate the spread
        self.spread = (self.dataf1 - self.dataf2).to_frame()  # calculate the spread of price

    def z_score_of_spread(self):  # indicator that according to Z-score base on the spread of price
        self.calculate_spread()
        cummean = self.spread.rolling(self.n).mean()  # calculate the expectation
        cumsigma = self.spread.rolling(self.n).std()  # calculate the standard deviation
        z_score_spread_n = pd.DataFrame((self.spread - cummean) / cumsigma)
        z_score_spread_n.columns = ["past " + str(self.n) + " days z_score_spread"]
        return z_score_spread_n

    def z_score_of_ratio(self):  # indicator that according to Z-score base on the ratio of price
        self.calculate_ratio()
        cummean = self.price_ratio.rolling(self.n).mean()  # calculate the expectation
        cumsigma = self.price_ratio.rolling(self.n).std()  # calculate the standard deviation
        z_score_ratio_n = pd.DataFrame((self.price_ratio - cummean) / cumsigma)
        z_score_ratio_n.columns = ["past " + str(self.n) + " days z_score_ratio"]
        return z_score_ratio_n

    def spread_MA(self):  # indicator that according to the trend of spread
        self.calculate_spread()
        spread_ma_n = self.spread.rolling(self.n).mean()
        spread_ma_n.columns = ['past ' + str(self.n) + 'days spread MA']
        return spread_ma_n

    def momentum(self):  # calculate momentum by (spread-MA)/MA
        MA = (self.spread_MA()).dropna()
        self.calculate_spread()
        data_t = pd.concat([MA, self.spread], join = 'inner', axis = 1)
        momentum_n = (data_t.iloc[:, 1] - data_t.iloc[:, 0]) / data_t.iloc[:, 0]
        momentum_n = momentum_n.to_frame()
        momentum_n.columns = ['past ' + str(self.n) + ' days momentum']
        return momentum_n

    def rolling_regression(self):  # indicator that according to the rolling regression
        model = PandasRollingOLS(self.dataf2, self.dataf1,window=self.n)
        result = (model.predicted).to_frame()
        y_ = result.groupby(level=0).last()
        y_.columns = ['second future rollingols '+str(self.n) + ' fitted value']
        return y_  # if the actual y is higher then this indicator, we can deem that y is underated

    def ind_standardize(self, df):
        # expanding standerdize
        cummean = df.expanding().mean()
        cumstd = df.expanding().std()
        indicator = (df - cummean) / cumstd
        indicator.dropna(inplace = True)
        return indicator

    def get_all_indicators(self):
        z_score_spread = self.z_score_of_spread()
        z_score_ratio = self.z_score_of_ratio()
        MA = self.spread_MA()
        momentum = self.momentum()
        fitted_y = self.rolling_regression()
        index = pd.concat([z_score_spread, z_score_ratio, MA, momentum,fitted_y], join = 'inner', axis = 1)
        index.dropna(inplace = True)
        result_index = self.ind_standardize(index)
        # drop the first trading year,and shift the indicator to next index:just for prediction
        result_index = ((result_index.shift(1)).dropna()).iloc[252:, :]
        self.indicators = result_index
        return result_index
