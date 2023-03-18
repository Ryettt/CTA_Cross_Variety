import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from sklearn import linear_model as lm
import matplotlib.ticker as ticker


class pair_future_data:
    # get close price : train_data and the remaining is for test data
    def __init__(self, rawdata, fcode1, fcode2, start_date, end_date):
        # rawdata: "maincontact.csv"
        # fcode1: wind code of future1 (string,eg:'Y.DCE')
        # fcode2: wind code of future2 (string)
        # start_date: data starting date (eg:'2010-01-25')
        # end_date: data ending date

        self.rawdata = rawdata
        self.fcode1 = fcode1
        self.fcode2 = fcode2
        self.start_date = start_date
        self.end_date = end_date

    def get_pair_data(self):
        data1 = (self.rawdata.loc[[self.fcode1, self.fcode2], :]).reset_index()
        data2 = (data1.pivot(index = 'TRADE_DT', columns = 'code', values = 'S_DQ_CLOSE').dropna()).sort_index(axis = 0)

        # train_data
        self.train_data = data2.loc[self.start_date:self.end_date, :]
        # remaining is test data
        self.test_data = data2.loc[self.end_date:, :]
        self.test_data = self.test_data.iloc[1:, :]

        # all_data
        self.all_data = data2

    def plot_prices(self):
        # plot the future close prices
        # fig, ax = plt.subplots()
        # le1 = ax.plot(self.train_data.iloc[:, 0], color = 'r', label = self.fcode1)
        # twin_ax = ax.twinx()
        # le2 = twin_ax.plot(self.train_data.iloc[:, 1], label = self.fcode2)
        # ax.legend(loc = 'upper left')
        # twin_ax.legend(loc = 'upper right')
        self.train_data.plot(secondary_y = self.train_data.columns[1])
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base = 100))
        plt.gcf().autofmt_xdate()
        plt.title('Price')
        plt.savefig("./image/" + "Price for " + self.fcode1 + " & " + self.fcode2 + ".png")
        plt.show()

    def plot_price_difference(self):
        model = lm.LinearRegression()
        model.fit(np.array(self.train_data.loc[:, self.fcode1]).reshape(-1, 1),
                  np.array(self.train_data.loc[:, self.fcode2]))
        ratio = model.coef_
        self.price_difference = self.train_data.loc[:, self.fcode2] - ratio * self.train_data.loc[:, self.fcode1]
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base = 200))
        plt.gcf().autofmt_xdate()
        plt.plot(self.price_difference)
        plt.title("Price Difference for " + self.fcode1 + " & " + self.fcode2)
        plt.savefig("./image/" + "Price Difference for " + self.fcode1 + " & " + self.fcode2 + ".png")
        plt.show()

    def get_price_corr(self):
        self.price_corr = np.corrcoef(self.train_data.iloc[:, 0], self.train_data.iloc[:, 1])[0, 1]
        print('The corr of the prices is: ')
        print(self.price_corr)
        return self.price_corr

    def contigration_test(self):
        # adfuller test
        print("Adfuller test of first difference of " + self.fcode1 + ":")
        print(adfuller(np.diff(self.train_data.iloc[:, 0])))
        print("-----------------------------------------------")

        print("Adfuller test of first difference of " + self.fcode2 + ":")
        print(adfuller(np.diff(self.train_data.iloc[:, 1])))
        print("-----------------------------------------------")

        print("Adfuller test of price_difference:")
        print(adfuller(self.price_difference))
        print("-----------------------------------------------")

        # cointegration_test
        print("cointegration_test: ")
        res = coint(self.train_data.iloc[:, 0], self.train_data.iloc[:, 1])
        print(res)
