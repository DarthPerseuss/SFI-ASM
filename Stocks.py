# For now we are assuming that all agent's can buy and sell as many holdings as they want.
# The sum of their shares would be the total number of shares in the market.
from statistics import mean
import random


class Stock:

    def __init__(self, num_shares, init_price, dividend_startvalue, rho, noise_sd, d_bar, interest_rate):
        self.numShares = num_shares
        self.price = init_price
        self.divTimeSeries = [dividend_startvalue]
        self.rho = rho
        self.arNoiseSD = noise_sd
        self.d_bar = d_bar
        self.priceTimeSeries = [init_price]
        self.marketCondition = [2 for _ in range(64)]
        self.r = interest_rate

    def advance_arprocess(self):
        """
        Implements AR(1) process for the series progression of dividends recursively.
        :return: appends next dividend to the time series
        """
        self.divTimeSeries.append(self.d_bar + self.rho *
                                  (self.divTimeSeries[-1] - self.d_bar) + random.gauss(mu=0, sigma=self.arNoiseSD))

    def set_price(self, value):
        self.priceTimeSeries = value
        self.price = value[-1]

    def calculate_ma(self, query, period, history=None):
        """
        Calculates dividend or price time series moving average for a given number of periods.
        :param query: string "dividend" or "price"
        :param period: number of time steps to look at
        :param history: how many windows of MA to look back into
        :return: n-period Moving average array or single entry
        """
        if query != "dividend" and query != "price":
            raise ValueError("Query can be only 'dividend' or 'price'")

        if history is not None:
            if query == "dividend":
                return mean(self.divTimeSeries[-period-1-history:-1-history])
            else:
                return mean(self.priceTimeSeries[-period - 1 - history:-1 - history])
        else:
            if query == "dividend":
                ma = []
                for ind in range(len(self.divTimeSeries) - period):
                    ma.append(mean(self.divTimeSeries[ind:period+ind]))
                return ma
            else:
                ma = []
                for ind in range(len(self.priceTimeSeries) - period):
                    ma.append(mean(self.priceTimeSeries[ind:period + ind]))
                return ma

    # We can start checking market condition after timestep 500
    def determine_marketcondition(self):
        """
        This function checks 64 market conditions that are given in the
        Java implementation of the code and checks if they evaluate to
        True or False at each time step. Thus the function should be run
        at every time step to give the current market condition.
        :return: updated market condition
        """

        if len(self.divTimeSeries) >= 10:
            for ind, dummy in enumerate([0.6, 0.8, 0.9, 1.0, 1.1, 1.12, 1.4]):
                if self.divTimeSeries[-1] / mean(self.divTimeSeries) >= dummy:
                    self.marketCondition[ind] = 1
                else:
                    self.marketCondition[ind] = 0

            for ind, dummy in enumerate([0.25, 0.5, 0.75, 0.875, 0.95, 1.0, 1.125]):
                if self.price * self.r / self.divTimeSeries[-1] >= dummy:
                    self.marketCondition[ind + 7] = 1
                else:
                    self.marketCondition[ind + 7] = 0

            for ind, dummy in enumerate([-1, -2, -3, -4]):
                # if the length of the time series is 2 you want to check if the price after the
                # price setting went up or not. In other words, we want to neglect the initial price,
                # hence addition by 1 (we are counting from 1 if we are counting backwards in the list,
                # 1-indexing or more precisely -1-indexing) + 1 (neglecting the initial price). Thus, we
                # add by 2.
                if len(self.divTimeSeries) <= abs(dummy) + 2:
                    self.marketCondition[ind + 14] = 2
                else:
                    if self.divTimeSeries[dummy] >= self.divTimeSeries[dummy - 1]:
                        self.marketCondition[ind + 14] = 1
                    else:
                        self.marketCondition[ind + 14] = 0
            for ind, dummy in enumerate([5, 10, 100, 500]):
                # We add by 1 because we want to be able to compare the moving average of two periods,
                # otherwise we don't care.
                if len(self.divTimeSeries) < dummy + 1:
                    self.marketCondition[ind + 18] = 2
                else:
                    if self.calculate_ma(query='dividend', period=dummy, history=1) < \
                            self.calculate_ma(query='dividend', period=dummy, history=0):
                        self.marketCondition[ind + 18] = 1
                    else:
                        self.marketCondition[ind + 18] = 0

            for ind, dummy in enumerate([5, 10, 100, 500]):
                if len(self.divTimeSeries) < dummy:
                    self.marketCondition[ind + 22] = 2
                else:
                    if self.divTimeSeries[-1] > self.calculate_ma(query='dividend', period=dummy, history=0):
                        self.marketCondition[ind + 22] = 1
                    else:
                        self.marketCondition[ind + 22] = 0

            for ind, dummy in enumerate([10, 100, 500]):
                if len(self.divTimeSeries) < dummy:
                    self.marketCondition[ind + 26] = 2
                else:
                    if self.calculate_ma(query='dividend', period=5, history=0) > \
                            self.calculate_ma(query='dividend', period=dummy, history=0):
                        self.marketCondition[ind+26] = 1
                    else:
                        self.marketCondition[ind+26] = 0

            for ind, dummy in enumerate([100, 500]):
                if len(self.divTimeSeries) < dummy:
                    self.marketCondition[ind+29] = 2
                else:
                    if self.calculate_ma(query='dividend', period=10, history=0) > \
                            self.calculate_ma(query='dividend', period=dummy, history=0):
                        self.marketCondition[ind+29] = 1
                    else:
                        self.marketCondition[ind+29] = 0

            if len(self.divTimeSeries) < 500:
                self.marketCondition[31] = 2
            else:
                if self.calculate_ma(query='dividend', period=100, history=0) > \
                        self.calculate_ma(query='dividend', period=500, history=0):
                    self.marketCondition[31] = 1
                else:
                    self.marketCondition[31] = 0

            for ind, dummy in enumerate([0.25, 0.5, 0.75, 0.875, 1.0, 1.125, 1.25]):
                if self.price / mean(self.priceTimeSeries) > dummy:
                    self.marketCondition[ind+32] = 1
                else:
                    self.marketCondition[ind+32] = 0

            for ind, dummy in enumerate([-1, -2, -3, -4]):
                if len(self.divTimeSeries) <= abs(dummy) + 2:
                    self.marketCondition[ind + 39] = 2
                else:
                    if self.priceTimeSeries[dummy] >= self.priceTimeSeries[dummy-1]:
                        self.marketCondition[ind + 39] = 1
                    else:
                        self.marketCondition[ind + 39] = 0

            for ind, dummy in enumerate([5, 10, 20, 100, 500]):
                if len(self.divTimeSeries) < dummy + 1:
                    self.marketCondition[ind + 44] = 2
                else:
                    if self.calculate_ma(query='price', period=dummy, history=1) < \
                            self.calculate_ma(query='price', period=dummy, history=0):
                        self.marketCondition[ind + 44] = 1
                    else:
                        self.marketCondition[ind + 44] = 0

            for ind, dummy in enumerate([5, 10, 20, 100, 500]):
                if len(self.divTimeSeries) < dummy:
                    self.marketCondition[ind + 49] = 2
                else:
                    if self.price > self.calculate_ma(query='price', period=dummy, history=0):
                        self.marketCondition[ind + 49] = 1
                    else:
                        self.marketCondition[ind + 49] = 0

            for ind, dummy in enumerate([10, 20, 100, 500]):
                if len(self.divTimeSeries) < dummy:
                    self.marketCondition[ind + 54] = 2
                else:
                    if self.calculate_ma(query='price', period=5, history=0) > \
                            self.calculate_ma(query='price', period=dummy, history=0):
                        self.marketCondition[ind + 54] = 1
                    else:
                        self.marketCondition[ind + 54] = 0

            for ind, dummy in enumerate([20, 100, 500]):
                if len(self.divTimeSeries) < dummy:
                    self.marketCondition[ind + 58] = 2
                else:
                    if self.calculate_ma(query='price', period=10, history=0) > \
                            self.calculate_ma(query='price', period=dummy, history=0):
                        self.marketCondition[ind + 58] = 1
                    else:
                        self.marketCondition[ind + 58] = 0

            for ind, dummy in enumerate([100, 500]):
                if len(self.divTimeSeries) < dummy:
                    self.marketCondition[ind + 61] = 2
                else:
                    if self.calculate_ma(query='price', period=20, history=0) > \
                            self.calculate_ma(query='price', period=dummy, history=0):
                        self.marketCondition[ind + 61] = 1
                    else:
                        self.marketCondition[ind + 61] = 0

            if len(self.divTimeSeries) < 500:
                self.marketCondition[63] = 2
            else:
                if self.calculate_ma(query='price', period=100, history=0) > \
                        self.calculate_ma(query='price', period=500, history=0):
                    self.marketCondition[63] = 1
                else:
                    self.marketCondition[63] = 0
