import random
from sklearn import linear_model, metrics
import numpy as np
from sfiasm.agents import TrendRegressor
from typing import Union


class BitCondition:

    @classmethod
    def return_queries(cls):
        print('original SFI-ASM; \n extended SFI-ASM; \n extended technical; \n original technical \n;'
              'original fundamental')

    @classmethod
    def predefined_rulesets(cls, query):
        if query == 'original SFI-ASM':
            return 7, 8, 9, 10, 11, 12, 13, 49, 50, 51, 52, 53
        elif query == 'extended SFI-ASM':
            return tuple(range(64))
        elif query == 'extended technical':
            return tuple(range(32, 64))
        elif query == 'original technical':
            return 49, 50, 51, 52, 53
        elif query == 'original fundamental':
            return 7, 8, 9, 10, 11, 12, 13
        else:
            raise NotImplementedError(f'Query {query} is not implemented.')

    def __init__(self, key):
        self.key = key
        self.value = 0

    def __eq__(self, other):
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        return f'BitCondition({str(self.key)}, {str(self.value)})'

    def assign_boolean(self, price_series: np.ndarray = None, div_series: np.ndarray = None, d_bar=None,
                       p_bar=None, interest_rate=None, moving_averages: dict = None):

        if self.key == 0:
            if div_series[-1] / d_bar > 0.6:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 1:
            if div_series[-1] / d_bar > 0.8:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 2:
            if div_series[-1] / d_bar > 0.9:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 3:
            if div_series[-1] / d_bar > 1.:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 4:
            if div_series[-1] / d_bar > 1.1:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 5:
            if div_series[-1] / d_bar > 1.12:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 6:
            if div_series[-1] / d_bar > 1.4:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 7:
            if price_series[-1] * interest_rate / div_series[-2] > 0.25:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 8:
            if price_series[-1] * interest_rate / div_series[-2] > 0.5:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 9:
            if price_series[-1] * interest_rate / div_series[-2] > 0.75:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 10:
            if price_series[-1] * interest_rate / div_series[-2] > 0.875:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 11:
            if price_series[-1] * interest_rate / div_series[-2] > 0.95:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 12:
            if price_series[-1] * interest_rate / div_series[-2] > 1.:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 13:
            if price_series[-1] * interest_rate / div_series[-2] > 1.125:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 14:
            if div_series[-1] > div_series[-2]:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 15:
            if div_series[-2] > div_series[-3]:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 16:
            if div_series[-3] > div_series[-4]:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 17:
            if div_series[-4] > div_series[-5]:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 18:
            # dma_5 = np.mean(div_series[-5:])
            previous_MA = moving_averages['dma_5'] - ((div_series[-1] - div_series[-6]) / 5)
            if previous_MA < moving_averages['dma_5']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 19:
            # dma_10 = np.mean(div_series[-10:])
            previous_MA = moving_averages['dma_10'] - ((div_series[-1] - div_series[-11]) / 10)
            if previous_MA < moving_averages['dma_10']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 20:
            # dma_100 = np.mean(div_series[-100:])
            previous_MA = moving_averages['dma_100'] - ((div_series[-1] - div_series[-101]) / 5)
            if previous_MA < moving_averages['dma_100']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 21:
            # dma_500 = np.mean(div_series[-500:])
            previous_MA = moving_averages['dma_500'] - ((div_series[-1] - div_series[-501]) / 5)
            if previous_MA < moving_averages['dma_500']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 22:
            if div_series[-1] > moving_averages['dma_5']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 23:
            if div_series[-1] > moving_averages['dma_10']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 24:
            if div_series[-1] > moving_averages['dma_100']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 25:
            if div_series[-1] > moving_averages['dma_500']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 26:
            if moving_averages['dma_5'] > moving_averages['dma_10']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 27:
            if moving_averages['dma_5'] > moving_averages['dma_100']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 28:
            if moving_averages['dma_5'] > moving_averages['dma_500']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 29:
            if moving_averages['dma_10'] > moving_averages['dma_100']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 30:
            if moving_averages['dma_10'] > moving_averages['dma_500']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 31:
            if moving_averages['dma_100'] > moving_averages['dma_500']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 32:
            if price_series[-1] / p_bar > 0.25:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 33:
            if price_series[-1] / p_bar > 0.5:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 34:
            if price_series[-1] / p_bar > 0.75:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 35:
            if price_series[-1] / p_bar > 0.875:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 36:
            if price_series[-1] / p_bar > 1.:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 37:
            if price_series[-1] / p_bar > 1.125:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 38:
            if price_series[-1] / p_bar > 1.25:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 39:
            if price_series[-1] > price_series[-2]:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 40:
            if price_series[-2] > price_series[-3]:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 41:
            if price_series[-3] > price_series[-4]:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 42:
            if price_series[-4] > price_series[-5]:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 43:
            if price_series[-5] > price_series[-6]:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 44:
            # pma_5 = np.mean(price_series[-5:])
            previous_MA = moving_averages['pma_5'] - ((price_series[-1] - price_series[-6]) / 5)
            if previous_MA < moving_averages['pma_5']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 45:
            # pma_10 = np.mean(price_series[-10:])
            previous_MA = moving_averages['pma_10'] - ((price_series[-1] - price_series[-11]) / 10)
            if previous_MA < moving_averages['pma_10']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 46:
            # pma_20 = np.mean(price_series[-20:])
            previous_MA = moving_averages['pma_20'] - ((price_series[-1] - price_series[-21]) / 20)
            if previous_MA < moving_averages['pma_20']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 47:
            # pma_100 = np.mean(price_series[-100:])
            previous_MA = moving_averages['pma_100'] - ((price_series[-1] - price_series[-101]) / 100)
            if previous_MA < moving_averages['pma_100']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 48:
            # pma_500 = np.mean(price_series[-500:])
            previous_MA = moving_averages['pma_500'] - ((price_series[-1] - price_series[-501]) / 500)
            if previous_MA < moving_averages['pma_500']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 49:
            if price_series[-1] > moving_averages['pma_5']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 50:
            if price_series[-1] > moving_averages['pma_10']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 51:
            if price_series[-1] > moving_averages['pma_20']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 52:
            if price_series[-1] > moving_averages['pma_100']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 53:
            if price_series[-1] > moving_averages['pma_500']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 54:
            if moving_averages['pma_5'] > moving_averages['pma_10']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 55:
            if moving_averages['pma_5'] > moving_averages['pma_20']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 56:
            if moving_averages['pma_5'] > moving_averages['pma_100']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 57:
            if moving_averages['pma_5'] > moving_averages['pma_500']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 58:
            if moving_averages['pma_10'] > moving_averages['pma_20']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 59:
            if moving_averages['pma_10'] > moving_averages['pma_100']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 60:
            if moving_averages['pma_10'] > moving_averages['pma_500']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 61:
            if moving_averages['pma_20'] > moving_averages['pma_100']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 62:
            if moving_averages['pma_20'] > moving_averages['pma_500']:
                self.value = 1
            else:
                self.value = 0

        elif self.key == 63:
            if moving_averages['pma_100'] > moving_averages['pma_500']:
                self.value = 1
            else:
                self.value = 0


class Market:

    def __init__(self, init_price, dividend_startvalue, rho, noise_sd, d_bar,
                 interest_rate, *, nconditions=None, bitkeys: Union[list, tuple] = None):

        self.price = init_price
        self.divTimeSeries = [dividend_startvalue]
        self.dividend = dividend_startvalue
        self.rho = rho
        self.arNoiseSD = noise_sd
        self.d_bar = d_bar
        self.priceTimeSeries = [init_price]

        if nconditions is not None:
            if not (nconditions == 39 or nconditions == 12 or nconditions == 32 or nconditions == 64):
                raise NotImplementedError

        if bitkeys is not None:
            self._bitkeys = bitkeys

            self._is_dma5 = any(x in self._bitkeys for x in [18, 22, 26, 27, 28])
            self._is_dma10 = any(x in self._bitkeys for x in [19, 23, 26, 29, 30])
            self._is_dma100 = any(x in self._bitkeys for x in [20, 24, 27, 29, 31])
            self._is_dma500 = any(x in self._bitkeys for x in [21, 25, 28, 30, 31])

            self._is_pma5 = any(x in self._bitkeys for x in [44, 49, 54, 55, 56, 57])
            self._is_pma10 = any(x in self._bitkeys for x in [45, 50, 54, 58, 59, 61])
            self._is_pma20 = any(x in self._bitkeys for x in [46, 51, 55, 58, 61, 62])
            self._is_pma100 = any(x in self._bitkeys for x in [47, 52, 56, 59, 61, 63])
            self._is_pma500 = any(x in self._bitkeys for x in [48, 53, 57, 60, 62, 63])

            self.marketCondition = {x: BitCondition(x) for x in self._bitkeys}
        # else:
        #     self.marketCondition = [2 for _ in range(nconditions)]

        self.r = interest_rate
        self.profitperunit = 0
        self.currentTime = 0
        self.divandprice_trend_conditions = [2 for _ in range(32)]

    def change_conditionstring(self, n):
        """

        :param n: should be a number that is implemented for market conditions
        :return:
        """

        self.marketCondition = [2 for _ in range(n)]

    def advance_arprocess(self):
        """
        Implements AR(1) process for the series progression of dividends recursively.
        :return: appends next dividend to the time series
        """
        # The dividend can be only paid in dollars and cents
        dividend = self.d_bar + self.rho * (self.dividend - self.d_bar) + \
                        random.gauss(mu=0, sigma=self.arNoiseSD)
        if dividend < 0.001:
            dividend = 0.001
        if dividend > 100:
            dividend = 100

        self.dividend = dividend
        self.divTimeSeries.append(self.dividend)

    def copy_arprocess(self, value):
        self.dividend = value
        self.divTimeSeries.append(value)

    def set_price(self, value):
        self.priceTimeSeries.append(value)
        self.price = value
        self.profitperunit = self.price - self.priceTimeSeries[-2] + self.dividend

    def calculate_ma(self, query, period, history=None):
        """
        Calculates dividend or price time series moving average for a given number of periods. if history
        is not given then the MA is calculated for th whole series.

        :param query: string "dividend" or "price"
        :param period: number of time steps to look at
        :param history: how many windows of MA to look back into
        :return: n-period Moving average array or single entry
        """
        if query != "dividend" and query != "price":
            raise ValueError("Query can be only 'dividend' or 'price'")

        if history is not None:
            # When using negative indexing a slice is 1-off indexed so list[-5:-1] denotes indices from
            # -5 to -2
            if query == "dividend":
                if history != 0:
                    return np.mean(self.divTimeSeries[-period - history: - history])
                else:
                    return np.mean(self.divTimeSeries[-period:])
            elif query == 'price':
                if history != 0:
                    return np.mean(self.priceTimeSeries[-period - history: - history])
                else:
                    return np.mean(self.priceTimeSeries[-period:])
        else:
            if query == "dividend":
                ma = []
                for ind in range(len(self.divTimeSeries) - period):
                    ma.append(np.mean(self.divTimeSeries[ind:period + ind]))
                return ma
            elif query == 'price':
                ma = []
                for ind in range(len(self.priceTimeSeries) - period):
                    ma.append(np.mean(self.priceTimeSeries[ind:period + ind]))
                return ma

    def linreg_universaltrend(self, trendfollowers, tau):
        """

        :param trendfollowers:
        :param tau: time horizon
        :return: the forecast based on the trend, the slope of the regression line (parameter a),
        the error (variance) of the regression estimate
        """
        reg = linear_model.LinearRegression().fit(np.arange(tau).reshape(-1, 1), self.priceTimeSeries[-tau:])
        y_pred = reg.coef_ * self.priceTimeSeries[-tau:] + reg.intercept_
        variance = metrics.mean_squared_error(self.priceTimeSeries[-tau:], y_pred)

        if variance < TrendRegressor.minvar:
            variance = TrendRegressor.minvar

        for agent in trendfollowers:
            agent.pdcoeff = reg.coef_[0]
            agent.offset = reg.intercept_
            agent.variance = variance

    def determine_marketcondition_bitconditions(self, compute_all_MAs=False):

        _price_series = np.array(self.priceTimeSeries[-500:])
        _div_series = np.array(self.priceTimeSeries[-500:])

        _moving_averages = {'dma_5': None, 'dma_10': None, 'dma_100': None, 'dma_500': None,
                            'pma_5': None, 'pma_10': None, 'pma_20': None, 'pma_100': None, 'pma_500': None}

        if compute_all_MAs is False:
            if self._is_dma5:
                _moving_averages['dma_5'] = _div_series[-5:].mean()
            if self._is_dma10:
                _moving_averages['dma_10'] = _div_series[-10:].mean()
            if self._is_dma100:
                _moving_averages['dma_100'] = _div_series[-100:].mean()
            if self._is_dma500:
                _moving_averages['dma_500'] = _div_series[-500:].mean()

            if self._is_pma5:
                _moving_averages['pma_5'] = _price_series[-5:].mean()
            if self._is_pma10:
                _moving_averages['pma_10'] = _price_series[-10:].mean()
            if self._is_pma20:
                _moving_averages['pma_20'] = _price_series[-20:].mean()
            if self._is_pma100:
                _moving_averages['pma_100'] = _price_series[-100:].mean()
            if self._is_pma500:
                _moving_averages['pma_500'] = _price_series[-500:].mean()

        else:
            _moving_averages['dma_5'] = _div_series[-5:].mean()
            _moving_averages['dma_10'] = _div_series[-10:].mean()
            _moving_averages['dma_100'] = _div_series[-100:].mean()
            _moving_averages['dma_500'] = _div_series[-500:].mean()
            _moving_averages['pma_5'] = _price_series[-5:].mean()
            _moving_averages['pma_10'] = _price_series[-10:].mean()
            _moving_averages['pma_20'] = _price_series[-20:].mean()
            _moving_averages['pma_100'] = _price_series[-100:].mean()
            _moving_averages['pma_500'] = _price_series[-500:].mean()

        for k in self._bitkeys:
            self.marketCondition[k].assign_boolean(price_series=_price_series, div_series=_div_series, d_bar=self.d_bar,
                                                   interest_rate=self.r, moving_averages=_moving_averages)

    def determine_divandpricetrends(self):

        dma_5 = self.calculate_ma(query='dividend', period=5, history=0)
        dma_10 = self.calculate_ma(query='dividend', period=10, history=0)

        pma_5 = self.calculate_ma(query='price', period=5, history=0)
        pma_10 = self.calculate_ma(query='price', period=10, history=0)

        pma_20 = self.calculate_ma(query='price', period=20, history=0)
        dma_20 = self.calculate_ma(query='dividend', period=20, history=0)

        dma_100 = self.calculate_ma(query='dividend', period=100, history=0)
        pma_100 = self.calculate_ma(query='price', period=100, history=0)

        dma_500 = self.calculate_ma(query='dividend', period=500, history=0)
        pma_500 = self.calculate_ma(query='price', period=500, history=0)

        divplusprice_mean = np.mean(self.priceTimeSeries[-5000:]) + 10

        diviplusprice = self.price + self.dividend

        for ind, dummy in enumerate([0.25, 0.5, 0.75, 0.875, 1.0, 1.125, 1.25]):
            if diviplusprice / divplusprice_mean > dummy:
                self.marketCondition[ind] = 1
            else:
                self.marketCondition[ind] = 0

        for ind, dummy in enumerate([-1, -2, -3, -4, -5]):
            if self.priceTimeSeries[dummy] + self.divTimeSeries[dummy] > \
                    self.priceTimeSeries[dummy - 1] + self.divTimeSeries[dummy - 1]:
                self.marketCondition[ind + 7] = 1
            else:
                self.marketCondition[ind + 7] = 0

        for ind, ma in enumerate(zip([pma_5, pma_10, pma_20, pma_100, pma_500],
                                     [dma_5, dma_10, dma_20, dma_100, dma_500],
                                     [5, 10, 20, 100, 500])):
            previous_MA = ma[0] + ma[1] - ((diviplusprice - (self.priceTimeSeries[-ma[2] - 1] +
                                                             self.divTimeSeries[-ma[2] - 1])) / ma[2])
            if previous_MA < ma[0] + ma[1]:
                self.marketCondition[ind + 12] = 1
            else:
                self.marketCondition[ind + 12] = 0

        for ind, ma in enumerate(zip([pma_5, pma_10, pma_20, pma_100, pma_500],
                                      [dma_5, dma_10, dma_20, dma_100, dma_500])):
            if diviplusprice > ma[0] + ma[1]:
                self.marketCondition[ind + 17] = 1
            else:
                self.marketCondition[ind + 17] = 0

        for ind, ma in enumerate(zip([pma_10, pma_20, pma_100, pma_500],
                                     [dma_10, dma_20, dma_100, dma_500])):
            if pma_5 + dma_5 > ma[0] + ma[1]:
                self.marketCondition[ind + 22] = 1
            else:
                self.marketCondition[ind + 22] = 0

        for ind, ma in enumerate(zip([pma_20, pma_100, pma_500],
                                     [dma_20, dma_100, dma_500])):
            if pma_10 + dma_10 > ma[0] + ma[1]:
                self.marketCondition[ind + 26] = 1
            else:
                self.marketCondition[ind + 26] = 0

        for ind, ma in enumerate(zip([pma_100, pma_500],
                                     [dma_100, dma_500])):
            if pma_20 + dma_20 > ma[0] + ma[1]:
                self.marketCondition[ind + 29] = 1
            else:
                self.marketCondition[ind + 29] = 0

        if pma_100 + dma_100 > pma_500 + dma_500:
            self.marketCondition[31] = 1
        else:
            self.marketCondition[31] = 0

    def determine_divpricetrend_plus_original(self):

        dma_5 = self.calculate_ma(query='dividend', period=5, history=0)
        dma_10 = self.calculate_ma(query='dividend', period=10, history=0)

        pma_5 = self.calculate_ma(query='price', period=5, history=0)
        pma_10 = self.calculate_ma(query='price', period=10, history=0)

        pma_20 = self.calculate_ma(query='price', period=20, history=0)
        dma_20 = self.calculate_ma(query='dividend', period=20, history=0)

        dma_100 = self.calculate_ma(query='dividend', period=100, history=0)
        pma_100 = self.calculate_ma(query='price', period=100, history=0)

        dma_500 = self.calculate_ma(query='dividend', period=500, history=0)
        pma_500 = self.calculate_ma(query='price', period=500, history=0)

        divplusprice_mean = np.mean(self.priceTimeSeries[-5000:]) + 10
        diviplusprice = self.price + self.dividend

        for ind, dummy in enumerate([0.25, 0.5, 0.75, 0.875, 0.95, 1.0, 1.125]):
            if self.price * (self.r / self.divTimeSeries[-2]) > dummy:
                self.marketCondition[ind] = 1
            else:
                self.marketCondition[ind] = 0

        for ind, pma in enumerate([pma_5, pma_10, pma_20, pma_100, pma_500]):
            if self.price > pma:
                self.marketCondition[ind + 7] = 1
            else:
                self.marketCondition[ind + 7] = 0

        for ind, dummy in enumerate([0.25, 0.5, 0.75, 0.875, 1.0, 1.125, 1.25]):
            if diviplusprice / divplusprice_mean > dummy:
                self.marketCondition[ind + 12] = 1
            else:
                self.marketCondition[ind + 12] = 0

        for ind, dummy in enumerate([-1, -2, -3, -4, -5]):
            if self.priceTimeSeries[dummy] + self.divTimeSeries[dummy] > \
                    self.priceTimeSeries[dummy - 1] + self.divTimeSeries[dummy - 1]:
                self.marketCondition[ind + 19] = 1
            else:
                self.marketCondition[ind + 19] = 0

        for ind, ma in enumerate(zip([pma_5, pma_10, pma_20, pma_100, pma_500],
                                     [dma_5, dma_10, dma_20, dma_100, dma_500],
                                     [5, 10, 20, 100, 500])):
            previous_MA = ma[0] + ma[1] - ((diviplusprice - (self.priceTimeSeries[-ma[2] - 1] +
                                                             self.divTimeSeries[-ma[2] - 1])) / ma[2])
            if previous_MA < ma[0] + ma[1]:
                self.marketCondition[ind + 24] = 1
            else:
                self.marketCondition[ind + 24] = 0

        for ind, ma in enumerate(zip([pma_5, pma_10, pma_20, pma_100, pma_500],
                                      [dma_5, dma_10, dma_20, dma_100, dma_500])):
            if diviplusprice > ma[0] + ma[1]:
                self.marketCondition[ind + 29] = 1
            else:
                self.marketCondition[ind + 29] = 0

        for ind, ma in enumerate(zip([pma_10, pma_20, pma_100, pma_500],
                                     [dma_10, dma_20, dma_100, dma_500])):
            if pma_5 + dma_5 > ma[0] + ma[1]:
                self.marketCondition[ind + 34] = 1
            else:
                self.marketCondition[ind + 34] = 0

        for ind, ma in enumerate(zip([pma_20, pma_100, pma_500],
                                     [dma_20, dma_100, dma_500])):
            if pma_10 + dma_10 > ma[0] + ma[1]:
                self.marketCondition[ind + 38] = 1
            else:
                self.marketCondition[ind + 38] = 0

        for ind, ma in enumerate(zip([pma_100, pma_500],
                                     [dma_100, dma_500])):
            if pma_20 + dma_20 > ma[0] + ma[1]:
                self.marketCondition[ind + 41] = 1
            else:
                self.marketCondition[ind + 41] = 0

        if pma_100 + dma_100 > pma_500 + dma_500:
            self.marketCondition[43] = 1
        else:
            self.marketCondition[43] = 0