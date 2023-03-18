from Indicator_construction import *
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.ticker as ticker


class Get_weight:

    def __init__(self, fcode1, fcode2, train_merge_data, test_merge_data):
        # train_data : r_n and indicator merged data
        # test_data: r_n and indicator merged data
        self.fcode1 = fcode1
        self.fcode2 = fcode2
        self.train_merge_data = train_merge_data
        self.test_merge_data = test_merge_data

    def get_ic(self, r_n, indicator1):
        np.seterr(divide = 'ignore', invalid = 'ignore')
        if indicator1.shape[1] > 1:

            ic = indicator1.apply(lambda y: np.corrcoef(y, r_n, rowvar = False)[0, 1])
        else:
            ic = np.corrcoef(indicator1, r_n, rowvar = False)[0, 1]
        ic = ic.astype('object')
        return ic

    def index_efficiency(self, r, ind):
        # r:future n days return of spread portfolio
        # ind:indicator
        datam = pd.concat([r, ind], axis = 1, join = 'inner').dropna()
        r_n = datam.iloc[:, 0]
        indicator1 = datam.iloc[:, 1:]

        # calculate ic of the optimize outcome
        ic = self.get_ic(r_n, indicator1)
        if indicator1.shape[1] > 1:

            params = sm.OLS(r_n, sm.add_constant(indicator1)).fit().params[1:]
            tvalue = indicator1.apply(lambda y: sm.OLS(r_n, sm.add_constant(y)).fit().tvalues[1])
            r2 = indicator1.apply(lambda y: sm.OLS(r_n, sm.add_constant(y)).fit().rsquared_adj)
            index_e = pd.concat([params, tvalue, r2, ic], axis = 1)
            index_e.columns = ['params', 'tvalue', 'adj_r2', 'IC']
        else:
            # just one indicator
            model = sm.OLS(r_n, sm.add_constant(indicator1)).fit()
            params = model.params.iloc[1]
            tvalue = model.tvalues.iloc[1]
            r2 = model.rsquared_adj
            index_e = pd.DataFrame([params, tvalue, r2, ic], index = ['params', 'tvalue', 'adj_r2', 'IC'])

        return index_e

    def weight_max_ic(self, w, r_n, indicator):
        # get the objective function
        # w:weight(unkown)
        # r_n:future n days return of spread portfolio
        w.shape = (1, indicator.shape[1])
        indicator1 = ((w * indicator).sum(axis = 1)).to_frame()
        ic = self.get_ic(r_n, indicator1)
        self.gw = -abs(ic)
        return self.gw

    def get_weight(self):
        # gw: objective function
        r_n = self.train_merge_data.iloc[:, 0]
        indicator = self.train_merge_data.iloc[:, 1:]
        w0 = np.ones((1, indicator.shape[1])) * (1 / indicator.shape[1])
        cons = ({'type': 'eq',
                 'fun': lambda w: w.sum() - 1},)
        x_bounds = []
        for j in range(len(indicator.columns)):
            x_bounds.append([-1, 1])

        wm = minimize(self.weight_max_ic, w0, args = (r_n, indicator), bounds = x_bounds, constraints = cons)
        w = wm.x
        self.train_windicator = (w * indicator).sum(axis = 1)
        self.train_ind_e = self.index_efficiency(r_n, self.train_windicator)  # index_efficiency
        self.train_ind_e.columns = ['train']
        self.w = w
        self.train_windicator = self.train_windicator.to_frame()
        self.train_windicator.columns = ['weight_indicator']
        self.compare_res = pd.concat([r_n, self.train_windicator], axis = 1)
        return self.w

    def get_windicator_for_test_data(self):

        self.test_windicator = ((self.w * self.test_merge_data.iloc[:, 1:]).sum(axis = 1)).to_frame()
        self.test_windicator.columns = ['weight_indicator']
        self.test_ind_e = self.index_efficiency(self.test_merge_data.iloc[:, [0]],
                                                self.test_windicator)  # index_efficiency
        self.test_ind_e.columns = ['test']
        self.test_res = pd.concat([self.test_merge_data.iloc[:, 0], self.test_windicator], axis = 1)

    def get_ind_efficiency(self):
        return pd.concat([self.train_ind_e, self.test_ind_e], axis = 1)

    def get_weight_indicator_for_all(self):
        return pd.concat([self.train_windicator, self.test_windicator])

    def train_outcome_plot(self):
        self.compare_res.plot(secondary_y = self.compare_res.columns[1])
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base = 100))
        plt.gcf().autofmt_xdate()
        plt.title("Train_outcome for " + self.fcode1 + " & " + self.fcode2)
        plt.savefig("./image/" + "Train_outcome for " + self.fcode1 + " & " + self.fcode2 + ".png")
        plt.show()

    def test_outcome_plot(self):
        self.test_res.plot(secondary_y = self.test_res.columns[1])
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base = 100))
        plt.gcf().autofmt_xdate()
        plt.title("Test_outcome for " + self.fcode1 + " & " + self.fcode2)
        plt.savefig("./image/" + "Test_outcome for " + self.fcode1 + " & " + self.fcode2 + ".png")
        plt.show()
