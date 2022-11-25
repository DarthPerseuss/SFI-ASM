import random
from queue import Queue
import numpy as np


class StratRules:

    def __init__(self, variance=None, actvariance=None, strength=None, gene_length=None,
                 manual_condition=None, pdcoeff=None, offset=None, activecount=None, lastactive=None,
                 tag=None):
        """

        :param gene_length: number of conditions
        :param manual_condition: Used to write custom conditions for example
        after mutation and crossover
        :param pdcoeff: parameter a in the forecast
        :param offset: parameter b in the forecast
        :param variance: Used to set the accuracy of the rule manually
        if not None
        """

        # The condition strings are initiated such that they get # (don't care) symbols
        # with a probability of 0.8 and the other two with probabilities of 0.1 each.
        # Here 0 means false 1 means True and 2 means "dont'care".
        if gene_length is not None and manual_condition is None:
            # Original SFI-ASM probabilities are (0,1,#) -> (0.05, 0.05, 0.9)
            self.condition = random.choices([0, 1, 2], weights=[0.5, 0.5, 9], k=gene_length)

        if gene_length is not None and manual_condition is not None:
            raise ValueError("Gene length and manual condition can't be both set")

        if type(manual_condition) == list and gene_length is None:
            self.condition = manual_condition

        if type(manual_condition) != list and gene_length is None:
            raise TypeError("The agent's condition string should be represented as a list")

        # The paramaters a and b are initiated here, but changed in the Agent class
        # since they undergo crossovers. They are defined as properties.

        if pdcoeff is not None:
            self.pdcoeff = pdcoeff
        else:
            self.pdcoeff = random.uniform(0.7, 1.2)

        if offset is not None:
            self.offset = offset
        else:
            self.offset = random.uniform(-10, 19)

        # In the Obj-C code this is referred to as variance (sigma)
        if actvariance is None:
            self.actVar = 4.000212
        else:
            self.actVar = actvariance

        if variance is None:
            self.variance = 4.000212
        else:
            self.variance = variance

        # Used for trading only. In the obj-C code this is
        # referred to as actVar (active variance estimate)

        self.deviation = 0
        self.forecast = 0
        self.lForecast = 0

        self.condition_length = len(self.condition)

        # To identify the default rule
        if tag is not None:
            self.tag = tag
        else:
            self.tag = 'ga'

        if strength is None:
            self.strength = 0
        else:
            self.strength = strength

        if activecount is None:
            self.activeCount = 0
        else:
            self.activeCount = activecount
        # 10 is set as the maximum forecast error change it if you must.

        if lastactive is None:
            self.lastActive = 0
        else:
            self.lastActive = lastactive

    @property
    def specfactor(self):
        """
        The specfactor parameter is computed after each change in the rules.
        Based on the addition term given in the init method.
        :return: Number of specific symbols (|0| + |1|)
        """
        return 0.005 * (self.condition_length - self.condition.count(2))

    def update_strength(self):
        self.variance = self.actVar
        self.strength = 100 - (self.variance + self.specfactor)

    def update_forecast(self, price, dividend):
        """

        :param price: this period's market price
        :param dividend: this period's market dividend
        :return: Expected returns for next period
        """
        self.lForecast = self.forecast
        self.forecast = self.pdcoeff * (price + dividend) + self.offset
        if self.forecast < 0:
            self.forecast = 0

    def update_forecast_nodiv(self, price):
        self.lForecast = self.forecast
        self.forecast = self.pdcoeff * price + self.offset
        if self.forecast < 0:
            self.forecast = 0

    # The function below should be applied one time-step later than the actual forecast accuracy is made
    # We want to compare the actual price to the forecast the agents made based on the previous time step.
    # So the current price and dividend come from the market and the previous price and dividend determine
    # the agent's forecast accuracy.
    def update_actvar(self, ftarget, a, b):

        self.deviation = (ftarget - self.lForecast) ** 2

        # Error is bounded
        if self.deviation > 100:
            self.deviation = 100
        self.actVar = b * self.actVar + a * self.deviation

    def is_activated(self, market_condition):
        """
        Checks whether the market state and the agent's condition match.
        Should be run at each time step.
        :param market_condition: Taken from the actual market dynamics, in its string format
        :return: Truth value
        """
        for c in range(self.condition_length):
            if self.condition[c] != 2:
                if self.condition[c] != market_condition[c]:
                    return False
        self.activeCount += 1
        return True


class StratRulesProfit(StratRules):

    rule_fitness_measure = 'profit_actVar'

    def __init__(self, variance=None, actvariance=None, strength=None, gene_length=None,
                 manual_condition=None, pdcoeff=None, offset=None, activecount=None, lastactive=None, profit=None,
                 tag=None):
        super().__init__(variance, actvariance, strength, gene_length, manual_condition, pdcoeff, offset,
                         activecount, lastactive, tag)

        # if profit is None:
        #     if self.rule_fitness_measure == 'profit_actVar':
        #         self.rule_profit = self.strength + self.actVar + self.specfactor
        #     elif self.rule_fitness_measure == 'actVar':
        #         self.rule_profit = self.strength + self.specfactor
        # else:
        if profit is None:
            self.rule_profit = 0
        else:
            self.rule_profit = profit

        self.rule_holding = 1
        self.rule_cleared = Queue(maxsize=2)
        self.rule_cleared.put(0)
        self.previous_clearing_price = 100
        self.current_clearing_price = 0

    def update_demand_rule(self, interest_rate, trialprice, agent_risk, mincash, cash, minholding, maxdemand):
        # This only for technical trading
        forecast = self.pdcoeff * trialprice + self.offset

        if forecast > 0:
            rule_demand = -((trialprice * (1 + interest_rate) - forecast) /
                                 (self.variance * agent_risk) + self.rule_holding)
            rule_slope = (self.pdcoeff - 1 - interest_rate) / (self.variance * agent_risk)
        else:
            rule_demand = -(trialprice * (1 + interest_rate) / (self.variance * agent_risk) + self.rule_holding)
            rule_slope = (- 1 - interest_rate) / (self.variance * agent_risk)

        # Restrict trading volume
        if rule_demand > maxdemand:
            rule_demand = maxdemand
            rule_slope = 0

        elif rule_demand < -maxdemand:
            rule_demand = -maxdemand
            rule_slope = 0

        if rule_demand > 0:
            if rule_demand * trialprice > (cash - mincash):
                if cash - mincash > 0:
                    rule_demand = (cash - mincash) / trialprice
                    rule_slope = -rule_demand / trialprice
                else:
                    rule_demand = 0
                    rule_slope = 0
        elif rule_demand < 0.0 and rule_demand + self.rule_holding < minholding:
            rule_demand = minholding - self.rule_holding
            rule_slope = 0
        return rule_demand, rule_slope

    def update_strength(self):

        self.variance = self.actVar

        if self.rule_fitness_measure == 'profit':
            self.strength = self.rule_profit - self.specfactor
        elif self.rule_fitness_measure == 'profit_actVar':
            self.strength = self.rule_profit - self.actVar - self.specfactor
