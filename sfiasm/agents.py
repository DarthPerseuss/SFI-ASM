import random
import copy
from operator import attrgetter
import networkx as nx
import numpy as np
from sklearn import linear_model
from sfiasm.strategyrules import StratRules, StratRulesProfit
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from typing import Union
import math


class Agent:
    _mincash = -1000
    # Short selling constraint
    _minholding = -5

    def __init__(self, init_holding, initialcash, risk):
        self.holding = init_holding
        self.risk = risk
        self.cash = initialcash
        self.profit = 0
        self.demand = 0.
        self.slope = 0.
        self.forecast = 0
        self.variance = 1
        self._maxdemand = 10

        # The number should be the initial price of the stock
        self.wealth = self.cash + 100.

    def adjust_holding(self, value):
        self.holding += value
        # minimum holding
        if self.holding < self._minholding:
            self.holding = self._minholding

    def adjust_cash(self, value):
        self.cash += value

        if self.cash < self._mincash:
            self.cash = self._mincash

    def creditearnings_paytaxes(self, dividend, price, intrate):
        if self.holding > 0:
            self.cash -= (price * intrate - dividend) * self.holding

        if self.cash < Agent._mincash:
            self.cash = Agent._mincash

        # Update wealth
        self.wealth = self.cash + price * self.holding

    def update_demand(self, trialprice, dividend, interest_rate, *args, **kwargs):
        pass

    def constrain_demand(self, trialprice):

        if self.demand > 0:
            if self.demand * trialprice > (self.cash - self._mincash):
                if self.cash - self._mincash > 0:
                    self.demand = (self.cash - self._mincash) / trialprice
                    self.slope = -self.demand / trialprice
                else:
                    self.demand = 0
                    self.slope = 0
                    if self.cash == self._mincash:
                        self.cash += 2000
        elif self.demand < 0.0 and self.demand + self.holding < self._minholding:
            self.demand = self._minholding - self.holding
            self.slope = 0.0


class GAagent(Agent):
    longtime = 4000
    maxdev = 100
    # Original SFI version uses 5
    mincount = 2
    init_var = 4.000212

    def __init__(self, init_holding, initialcash, num_strategies, theta, rule_keys,
                 risk, is_learning=True, ruleparam_range: Union[tuple, list, np.ndarray] = None):
        """
        In my design here, I have separated the default forecast from other forecasting
        rules. So pay attention to updating it.
        :param init_holding:
        :param initialcash:
        :param num_strategies:
        :param theta:
        :param rule_keys:
        :param risk:
        :param is_learning: bool
        """

        super().__init__(init_holding, initialcash, risk)
        self.tag = 'ga'  # tag to denote the profile of the agent
        self.gene_length = len(rule_keys)
        self.numStrategies = num_strategies - 1
        # if not (gene_length == 12 or gene_length == 64 or gene_length == 32):
        #     raise ValueError("12, 32, and 64 are accepted gene lengths")

        if ruleparam_range is None:
            self._maxpdcoeff = 1.2
            self._minpdcoeff = 0.7
            self._maxoffset = 19
            self._minoffset = -10

            self.strategies = [StratRules(variance=GAagent.init_var,
                                          gene_length=self.gene_length)
                               for _ in range(self.numStrategies)]
            self.defaultRule = StratRules(variance=GAagent.init_var,
                                          activecount=GAagent.mincount,
                                          manual_condition=[2 for _ in range(self.gene_length)], tag='default')
        else:
            self._maxpdcoeff = 0.95 + ruleparam_range[0]
            self._minpdcoeff = 0.95 - ruleparam_range[0]
            self._maxoffset = 4.5 + ruleparam_range[1]
            self._minoffset = 4.5 - ruleparam_range[1]

            self.strategies = [StratRules(variance=GAagent.init_var,
                                          gene_length=self.gene_length,
                                          pdcoeff=random.uniform(self._minpdcoeff, self._maxpdcoeff),
                                          offset=random.uniform(self._minoffset, self._maxoffset))
                               for _ in range(self.numStrategies)]

            self.defaultRule = StratRules(variance=GAagent.init_var,
                                          activecount=GAagent.mincount,
                                          pdcoeff=random.uniform(self._minpdcoeff, self._maxpdcoeff),
                                          offset=random.uniform(self._minoffset, self._maxoffset),
                                          manual_condition=[2 for _ in range(self.gene_length)], tag='default')

        # theta measures how much the agents look into the past
        self.theta = theta
        self.is_learning = is_learning
        self.oldActiveList = list()
        self.activeList = list()

        self.rule_keys = rule_keys

        # for demands
        self.chosen_pdcoeff = 0
        self.chosen_offset = 0
        self.chosen_divisor = 0

        self._nnew = 20
        # Used if a pool of rejections is made
        self._nreject = 2 * self._nnew
        self._newrules = []

        self.variance = GAagent.init_var
        self.forecast = self.lForecast = 0

        self.technical_frac = 0
        self.fundamental_frac = 0
        self.ga_activated = False

    @property
    def all_rules(self):
        return self.strategies + [self.defaultRule]

    def determine_activated(self, market):
        """
        Method to determine the rule that will be selected to compute the demand
        :return: selected rule j
        """

        # dump old active list
        self.oldActiveList = []
        # fill old list with previous steps active forecasts
        self.oldActiveList += self.activeList

        # Check
        for old in self.oldActiveList:
            old.lForecast = old.forecast

        self.activeList = [self.defaultRule]
        self.defaultRule.lastActive += 1
        self.defaultRule.activeCount += 1

        currentTime = market.currentTime

        market_condition = [market.marketCondition[k].value for k in self.rule_keys]
        for rule in self.strategies:
            if rule.is_activated(market_condition=market_condition):
                self.activeList.append(rule)
                rule.lastActive = currentTime

    def prepare_for_trading(self, ga_probability, mutation_per_bit, crossover_rate,
                            market, removal_replacement_ratio=0.5):

        if market.currentTime > 1 / ga_probability:
            # Using median or mean doesn't make qualitative difference
            self.perform_ga(ga_probability, mutation_per_bit, crossover_rate,
                            market, removal_replacement_ratio=removal_replacement_ratio)
            self.ga_activated = True

        self.lForecast = self.forecast
        self.determine_activated(market)

        mincount = GAagent.mincount
        valid_actives = [active for active in self.activeList if active.activeCount >= mincount]
        nactive = len(valid_actives)

        try:
            bestForecast = min(valid_actives, key=attrgetter('actVar'))
        except ValueError:
            bestForecast = None

        # Meaning some rule is active
        if nactive != 0:
            self.chosen_pdcoeff = bestForecast.pdcoeff
            self.chosen_offset = bestForecast.offset
            forecastvar = bestForecast.variance

        self.chosen_divisor = self.risk * forecastvar

    def update_fcast(self, p_trial, dividend):
        self.forecast = (p_trial + dividend) * self.chosen_pdcoeff + self.chosen_offset

    def update_demand(self, p_trial, dividend, interest_rate, *args, **kwargs):
        """

        :param interest_rate:
        :param dividend:
        :param p_trial: price set by the specialist
        :return: desired demand for the agent along with the demand derivative and a dummy variable
        which is used in calculating the trial price.
        """

        self.update_fcast(p_trial, dividend)

        if self.forecast > 0:
            self.demand = -((p_trial * (1 + interest_rate) - self.forecast) / self.chosen_divisor + self.holding)
            self.slope = (self.chosen_pdcoeff - 1 - interest_rate) / self.chosen_divisor
        else:
            self.forecast = 0
            self.demand = -(p_trial * (1 + interest_rate) / self.chosen_divisor + self.holding)
            self.slope = (- 1 - interest_rate) / self.chosen_divisor

        # Restrict trading volume
        if self.demand > self._maxdemand:
            self.demand = self._maxdemand
            self.slope = 0

        elif self.demand < -self._maxdemand:
            self.demand = -self._maxdemand
            self.slope = 0

        self.constrain_demand(p_trial)

    def update_performance(self, market):
        a = 1 / self.theta
        b = 1 - a

        ftarget = market.price + market.dividend

        for rule in self.activeList:
            rule.update_forecast(market.price, market.dividend)

        if market.currentTime >= 1:
            for rule in self.oldActiveList:
                rule.update_actvar(ftarget, a, b)

    def average_strength(self):
        return np.mean([rule.strength for rule in self.strategies])

    def perform_ga(self, probability, mutation_per_bit, crossover_rate,
                   market, removal_replacement_ratio):

        """

        :param probability:
        :param mutation_per_bit:
        :param crossover_rate:
        :return:
        """

        if random.uniform(0, 1) < probability:

            current_time = market.currentTime

            avstrength = self.calculate_median()

            madv = self.calculate_madv()

            # Replace 20 worst rules at each invocation
            reject_list = sorted(self.strategies, key=attrgetter('strength'))[:self._nreject]

            while len(self._newrules) < self._nnew:
                parent1 = self.tournament()
                parent2 = None
                r = random.uniform(0, 1)

                if r < crossover_rate:
                    is_same_parent = True
                    while is_same_parent:
                        parent2 = self.tournament()
                        if parent1 != parent2:
                            is_same_parent = False

                    self.crossover(parent1, parent2, avstrength, current_time=current_time)
                else:
                    self.mutate(parent1, mutation_per_bit, avstrength, current_time=current_time)

            self.set_newrule_params(madv, avstrength)
            if random.uniform(0, 1) < removal_replacement_ratio:
                self.remove_worst()
                self.strategies += self._newrules
            else:
                self.replace_rules(reject_list)
            self._newrules = []
            self.generalize(avstrength, current_time)

    def tournament(self) -> Union[StratRulesProfit, StratRules]:

        active = False
        i = 0
        r1 = 0
        while not active and i < 50:
            r1 = random.randint(0, len(self.strategies) - 1)
            if self.strategies[r1].activeCount > 0:
                active = True
            i += 1
        candidate1 = self.strategies[r1]

        active = False
        i = 0
        r2 = 0
        while not active and i < 50:
            r2 = random.randint(0, len(self.strategies) - 1)
            while r2 == r1:
                r2 = random.randint(0, len(self.strategies) - 1)
            if self.strategies[r2].activeCount > 0:
                active = True
            i += 1
        candidate2 = self.strategies[r2]

        if candidate1.strength > candidate2.strength:
            return candidate1
        else:
            return candidate2

    def calculate_median(self):

        # The rules are updated here?
        for rule in self.all_rules:
            if rule.activeCount != 0:
                rule.update_strength()

        return np.median([rule.strength for rule in self.all_rules])

    def calculate_mean(self):

        for rule in self.all_rules:
            if rule.activeCount != 0:
                rule.variance = rule.actVar
                rule.strength = GAagent.maxdev - (rule.variance + rule.specfactor)

        return np.mean([rule.strength for rule in self.all_rules])

    def calculate_madv(self):

        meanv = ava = avb = sumc = 0.0
        all_rules = self.all_rules
        for rule in all_rules:
            varvalue = rule.variance
            meanv += varvalue
            if rule.activeCount > 0:
                if varvalue != 0:
                    sumc += 1 / varvalue
                    ava += rule.pdcoeff / varvalue
                    avb += rule.offset / varvalue

        meanv /= (self.numStrategies + 1)

        madv = sum([abs(rule.variance - meanv) for rule in all_rules]) / (self.numStrategies + 1)

        # Set rule 0 (alway all don t care) to inverse variance weight
        # of the forecast parameters.
        self.defaultRule.pdcoeff = ava / sumc
        self.defaultRule.offset = avb / sumc

        return madv

    def set_newrule_params(self, madv, medstrength):

        for rule in self._newrules:

            rule.actVar = GAagent.maxdev - (rule.strength + rule.specfactor)

            if rule.actVar < self.defaultRule.variance - madv:
                rule.actVar = self.defaultRule.variance - madv
                rule.strength = GAagent.maxdev - (rule.actVar + rule.specfactor)

            if rule.actVar <= 0:
                rule.actVar = GAagent.maxdev - (medstrength + rule.specfactor)
                rule.strength = medstrength

            rule.variance = rule.actVar

    def _add_newrules(self, offspring_strength, offspring_condition, offspring_pdcoeff,
                      offspring_offset, offspring_lastActive, offspring_activeCount=None):

        self._newrules.append(StratRules(strength=offspring_strength,
                                         manual_condition=offspring_condition,
                                         pdcoeff=offspring_pdcoeff,
                                         offset=offspring_offset,
                                         lastactive=offspring_lastActive,
                                         activecount=offspring_activeCount))

    def mutate(self, selected_parent, mutation_per_bit, avgstrength, current_time, avg_var=None):
        """
        The mutation of the forecasts is done in 3 ways as well. With probability 0.2,
        they are uniformly changed to a value within the permissible ranges of the parameters
        which is [0.7, 1.2] for the a parameter and[ −10.0, 19.0] for the b parameter. With
        probability 0.2, the current parameter value is uniformly distributed within ± 5%
        of its current value, and it is left unchanged for the remaining cases.

        :param avg_var:
        :param current_time:
        :param avgstrength:
        :param selected_parent:
        :param mutation_per_bit:
        :return: mutated offspring
        """
        if self.is_learning is True:
            offspring_condition = copy.deepcopy(selected_parent.condition)

            # Perform condition mutation. If slow store length functions in a variable
            # The weights preserve the average number of don't care (#) symbols, here
            # denoted as 2.
            bitchanged = False
            changed = False

            for i in range(len(offspring_condition)):
                if random.uniform(0, 1) <= mutation_per_bit:
                    if offspring_condition[i] == 0:
                        offspring_condition[i] = random.choices((1, 2), weights=[1, 2])[0]
                        changed = bitchanged = True
                    elif offspring_condition[i] == 1:
                        offspring_condition[i] = random.choices((0, 2), weights=[1, 2])[0]
                        changed = bitchanged = True
                    else:
                        offspring_condition[i] = random.choice((0, 1, 2))
                        if offspring_condition[i] != selected_parent.condition[i]:
                            changed = bitchanged = True

            # Perform the forecast mutation
            r_pdcoeff = random.uniform(0, 1)
            if r_pdcoeff <= 0.2:
                offspring_pdcoeff = random.uniform(self._minpdcoeff, self._maxpdcoeff)
                changed = True

            elif 0.2 < r_pdcoeff <= 0.4:
                changed = True
                offspring_pdcoeff = random.uniform(
                    selected_parent.pdcoeff - 0.05 * (self._maxpdcoeff - self._minpdcoeff),
                    selected_parent.pdcoeff + 0.05 * (self._maxpdcoeff - self._minpdcoeff))
                if offspring_pdcoeff > self._maxpdcoeff:
                    offspring_pdcoeff = self._maxpdcoeff
                elif offspring_pdcoeff < self._minpdcoeff:
                    offspring_pdcoeff = self._minpdcoeff

            else:
                offspring_pdcoeff = selected_parent.pdcoeff

            r_offset = random.uniform(0, 1)
            if r_offset <= 0.2:
                changed = True
                offspring_offset = random.uniform(self._minoffset, self._maxoffset)

            elif 0.2 < r_offset <= 0.4:
                changed = True
                offspring_offset = random.uniform(
                    selected_parent.offset - 0.05 * (self._maxoffset - self._minoffset),
                    selected_parent.offset + 0.05 * (self._maxoffset - self._minoffset))
                if offspring_offset > self._maxoffset:
                    offspring_offset = self._maxoffset
                elif offspring_offset < self._minoffset:
                    offspring_offset = self._minoffset

            else:
                offspring_offset = selected_parent.offset

            offspring_strength = selected_parent.strength
            offspring_lastActive = current_time

            # ** My version
            # if selected_parent.activeCount == 0 or (current_time - selected_parent.lastActive) > GAagent.longtime:
            #     offspring_strength = avgstrength
            #     if not bitchanged:
            #         offspring_lastActive = selected_parent.lastActive
            #
            # if bitchanged:
            #     offspring_strength = avgstrength

            if changed:
                if selected_parent.activeCount == 0 or (current_time - selected_parent.lastActive) > GAagent.longtime:
                    offspring_strength = avgstrength
                    if not bitchanged:
                        offspring_lastActive = selected_parent.lastActive
                else:
                    if bitchanged:
                        offspring_strength = avgstrength
                    else:
                        offspring_lastActive = selected_parent.lastActive

            self._add_newrules(offspring_strength, offspring_condition, offspring_pdcoeff,
                               offspring_offset, offspring_lastActive)

    def crossover(self, parent1, parent2, avgstrength, current_time):
        """
        Chooses two parents by tournament selection and performs uniform crossover between
        them to get the new conditions. Performs one of three tasks to perform the crossover
        for the forecasts with equal probability. One is to choose a parent and get both genes
        from that single parent. The other is to randomly choose each of the parents for each
        gene. The last method uses a weighted average of both parents to determine a and b
        parameters in the offspring.

        :return: crossovered offspring
        """
        if self.is_learning is True:
            # This line performs the uniform corssover
            offspring_condition = [random.choice((parent1.condition[i], parent2.condition[i]))
                                   for i in range(self.gene_length)]

            # This random number determines which strategy is used for the parameters of the offspring
            r = random.uniform(0, 1)
            if r < 1 / 3:
                chosen_parent = random.choice((parent1, parent2))
                crossed_pdcoeff = chosen_parent.pdcoeff
                crossed_offset = chosen_parent.offset

            elif 1 / 3 <= r < 2 / 3:
                chosenparent_for_param_a = random.choice((parent1, parent2))
                chosenparent_for_param_b = random.choice((parent1, parent2))
                crossed_pdcoeff = chosenparent_for_param_a.pdcoeff
                crossed_offset = chosenparent_for_param_b.offset

            else:
                weight1 = (1 / parent1.variance) / (1 / parent1.variance + 1 / parent2.variance)
                weight2 = 1 - weight1
                crossed_pdcoeff = weight1 * parent1.pdcoeff + \
                                  weight2 * parent2.pdcoeff
                crossed_offset = weight1 * parent1.offset + \
                                 weight2 * parent2.offset

            offspring_lastActive = current_time

            # if parent1.activeCount > parent2.activeCount:
            #     offspring_activeCount = parent1.activeCount
            # else:
            #     offspring_activeCount = parent2.activeCount

            offspring_activeCount = 0

            # This is following after Obj-C code
            if parent1.activeCount * parent2.activeCount == 0 or \
                    current_time - parent1.lastActive > GAagent.longtime or \
                    current_time - parent2.lastActive > GAagent.longtime:
                offspring_strength = avgstrength

            else:
                offspring_strength = (parent1.strength + parent2.strength) / 2

            self._add_newrules(offspring_strength, offspring_condition, crossed_pdcoeff,
                               crossed_offset, offspring_lastActive, offspring_activeCount)

    def replace_rules(self, rejectList):

        temp_list = rejectList
        for rule in self._newrules:
            candidate1 = random.randint(0, len(temp_list) - 1)
            candidate2 = None

            is_same_candidate = True
            while is_same_candidate:
                candidate2 = random.randint(0, len(temp_list) - 1)
                if candidate1 != candidate2:
                    is_same_candidate = False

            # Three versions to choose from randomly
            # r = random.uniform(0, 1)
            # if r < 1/2:
            similarity1 = sum(x == y for x, y in zip(rule.condition, temp_list[candidate1].condition))
            similarity2 = sum(x == y for x, y in zip(rule.condition, temp_list[candidate2].condition))
            if similarity1 > similarity2:
                # Using list.remove() messes up memory allocations
                del self.strategies[self.strategies.index(temp_list[candidate1])]
                del temp_list[candidate1]

            else:
                del self.strategies[self.strategies.index(temp_list[candidate2])]
                del temp_list[candidate2]

            # elif 1/3 <= r < 2/3:
            #     removed = random.choice((candidate1, candidate2))
            #     del self.strategies[self.strategies.index(temp_list[removed])]
            #     del temp_list[removed]
            #
            # else:
            #     if temp_list[candidate1].activeCount < temp_list[candidate2].activeCount:
            #         del self.strategies[self.strategies.index(temp_list[candidate1])]
            #         del temp_list[candidate1]
            #     else:
            #         del self.strategies[self.strategies.index(temp_list[candidate2])]
            #         del temp_list[candidate2]

            self.strategies.append(rule)

    def remove_worst(self):
        self.strategies = sorted(self.strategies, key=attrgetter('strength'))
        del self.strategies[0: self._nnew]

    def generalize(self, avgstrength, current_time):
        """
        Method that converts 1/4 of
         True or False symbols to
        don't care (#) symbols inside the condition part.
        :param current_time:
        :param avgstrength: median fitness in the agent's rule set to set
        the fitness of a generalized rule in case it has undergone generalization
        :return: generalized rule
        """
        for rule in self.strategies:
            if current_time - rule.lastActive > GAagent.longtime:
                care_indices = [ind for ind in range(len(rule.condition)) if rule.condition[ind] != 2]
                oneforth = math.ceil(len(care_indices) / 4)
                if oneforth < 1:
                    oneforth = 1
                chosen_oneforth = random.choices(care_indices, k=oneforth)

                for c_ind in chosen_oneforth:
                    rule.condition[c_ind] = 2
                    rule.activeCount = 0
                    rule.lastActive = current_time
                    rule.actVar = GAagent.maxdev - (avgstrength + rule.specfactor)
                    if rule.actVar <= 0:
                        rule.actVar = rule.variance

                    rule.strength = avgstrength

    @staticmethod
    # TODO this needs to be thought through on which strategies the agents are presenting.
    # For example is it going to be the set of strategies being activated for more timesteps
    # or are they the activated rules with best strength?
    def present_strategies(n, activated_rules):
        """
        Used when these agents are established as competent agents in a
        network to be imitated by other agents.
        :return: the n best strategies of the agent which are activated
        """
        try:
            return sorted(activated_rules, key=attrgetter('strength'), reverse=True)[0:n]
        except IndexError:
            return sorted(activated_rules, key=attrgetter('strength'), reverse=True)


class UnderinformedAgent(GAagent):
    #
    def __init__(self, init_holding, initialcash, num_strategies, theta,
                 gene_length, risk, divdend_error_var):
        super().__init__(init_holding, initialcash, num_strategies, theta,
                         gene_length, risk)

        self.dividend_error_var = divdend_error_var
        self.tag = 'under'
        self.error_term = 0

    # def update_errorterm(self):
    #     self.error_term = np.random.normal(0, self.dividend_error_var)
    #
    # def update_fcast(self, p_trial, dividend):
    #     # The error term is added
    #     self.forecast = (p_trial + dividend + self.error_term) \
    #                      * self.chosen_pdcoeff + self.chosen_offset

    def update_fcast(self, p_trial, dividend):
        # self.error_term = np.random.normal(0, self.dividend_error_var)
        np.random.seed(abs(int(math.modf(self.holding)[1] + 100 * math.modf(self.holding)[1])))
        # The error term is added
        self.forecast = (p_trial + dividend + np.random.normal(0, self.dividend_error_var)) \
                         * self.chosen_pdcoeff + self.chosen_offset

    def update_performance(self, market):
        a = 1 / self.theta
        b = 1 - a

        ftarget = market.price + market.dividend
        error_term = np.random.normal(0, self.dividend_error_var)
        for rule in self.activeList:
            rule.lForecast = rule.forecast
            rule.forecast = rule.pdcoeff * (ftarget + error_term) + rule.offset
            if rule.forecast < 0:
                rule.forecast = 0

        if market.currentTime >= 1:
            for rule in self.oldActiveList:
                rule.update_actvar(ftarget + error_term, a, b)


class ModifiedGA(GAagent):
    def __init__(self, init_holding,  initialcash, num_strategies, theta,
                 rule_keys, risk, ruleparam_range=None, ruleparam_center=None):
        super().__init__(init_holding, initialcash, num_strategies, theta, rule_keys, risk,
                         ruleparam_range=ruleparam_range)

        if ruleparam_range is None and ruleparam_center is None:
            self._maxpdcoeff = 1.2
            self._minpdcoeff = 0.7
            self._maxoffset = 19
            self._minoffset = -10

            self.strategies = [
                StratRules(variance=GAagent.init_var, gene_length=self.gene_length)
                for _ in range(self.numStrategies)]
            self.defaultRule = StratRules(variance=GAagent.init_var,
                                          activecount=GAagent.mincount,
                                          manual_condition=[2 for _ in range(self.gene_length)], tag='default')
        else:
            if ruleparam_center is None:
                self.pdcoef_center = 0.95
                self.offset_center = 4.5

                self._maxpdcoeff = self.pdcoef_center + ruleparam_range[0]
                self._minpdcoeff = self.pdcoef_center - ruleparam_range[0]
                self._maxoffset = self.offset_center + ruleparam_range[1]
                self._minoffset = self.offset_center - ruleparam_range[1]

            else:
                if ruleparam_center[0] is not None:
                    self.pdcoef_center = ruleparam_center[0]
                else:
                    self.pdcoef_center = 0.95

                if ruleparam_center[1] is not None:
                    self.offset_center = ruleparam_center[1]
                else:
                    self.offset_center = 4.5

                self._maxpdcoeff = self.pdcoef_center + ruleparam_range[0]
                self._minpdcoeff = self.pdcoef_center - ruleparam_range[0]
                self._maxoffset = self.offset_center + ruleparam_range[1]
                self._minoffset = self.offset_center - ruleparam_range[1]

            self.strategies = [
                StratRules(variance=GAagent.init_var, gene_length=self.gene_length,
                                 pdcoeff=random.uniform(self._minpdcoeff, self._maxpdcoeff),
                                 offset=random.uniform(self._minoffset, self._maxoffset))
                for _ in range(self.numStrategies)]

            self.defaultRule = StratRules(variance=GAagent.init_var,
                                          pdcoeff=random.uniform(self._minpdcoeff, self._maxpdcoeff),
                                          offset=random.uniform(self._minoffset, self._maxoffset),
                                          manual_condition=[2 for _ in range(self.gene_length)], tag='default')

    def mutate(self, selected_parent, mutation_per_bit, avgstrength, current_time, avg_var=None):

        if self.is_learning is True:
            offspring_condition = copy.deepcopy(selected_parent.condition)

            # Perform condition mutation. If slow store length functions in a variable
            # The weights preserve the average number of don't care (#) symbols, here
            # denoted as 2.

            bitchanged = False
            specificity = int(selected_parent.specfactor / 0.005)
            for i in range(len(offspring_condition)):
                if random.uniform(0, 1) <= mutation_per_bit:

                    if offspring_condition[i] == 0:
                        offspring_condition[i] = random.choices((1, 2), weights=[1, 2])[0]
                    bitchanged = True
                    if offspring_condition[i] == 2:
                        specificity -= 1

                    elif offspring_condition[i] == 1:
                        offspring_condition[i] = random.choices((0, 2), weights=[1, 2])[0]
                        bitchanged = True
                        if offspring_condition[i] == 2:
                            specificity -= 1

                    else:
                        offspring_condition[i] = random.choice((0, 1, 2))
                        if offspring_condition[i] != selected_parent.condition[i]:
                            bitchanged = True
                            specificity += 1

            if specificity == 0:
                offspring_condition[random.randint(0, self.gene_length - 1)] = random.choice((0, 1))
                bitchanged = True

            # Perform the forecast mutation
            r_pdcoeff = random.uniform(0, 1)
            if r_pdcoeff <= 0.2:
                offspring_pdcoeff = random.uniform(self._minpdcoeff, self._maxpdcoeff)

            elif 0.2 < r_pdcoeff <= 0.4:
                offspring_pdcoeff = random.uniform(
                    selected_parent.pdcoeff - 0.05 * (self._maxpdcoeff - self._minpdcoeff),
                    selected_parent.pdcoeff + 0.05 * (self._maxpdcoeff - self._minpdcoeff))
                if offspring_pdcoeff > self._maxpdcoeff:
                    offspring_pdcoeff = self._maxpdcoeff
                elif offspring_pdcoeff < self._minpdcoeff:
                    offspring_pdcoeff = self._minpdcoeff

            else:
                offspring_pdcoeff = selected_parent.pdcoeff

            r_offset = random.uniform(0, 1)
            if r_offset <= 0.2:
                offspring_offset = random.uniform(self._minoffset, self._maxoffset)

            elif 0.2 < r_offset <= 0.4:
                offspring_offset = random.uniform(
                    selected_parent.offset - 0.05 * (self._maxoffset - self._minoffset),
                    selected_parent.offset + 0.05 * (self._maxoffset - self._minoffset))
                if offspring_offset > self._maxoffset:
                    offspring_offset = self._maxoffset
                elif offspring_offset < self._minoffset:
                    offspring_offset = self._minoffset

            else:
                offspring_offset = selected_parent.offset

            offspring_strength = selected_parent.strength
            offspring_lastActive = current_time

            # ** My version
            if selected_parent.activeCount == 0 or (current_time - selected_parent.lastActive) > GAagent.longtime:
                offspring_strength = avgstrength
                if not bitchanged:
                    offspring_lastActive = selected_parent.lastActive

            if bitchanged:
                offspring_strength = avgstrength

            self._newrules.append(StratRules(strength=offspring_strength,
                                             manual_condition=offspring_condition,
                                             pdcoeff=offspring_pdcoeff,
                                             offset=offspring_offset,
                                             lastactive=offspring_lastActive))

    def generalize(self, avgstrength, current_time):
        for rule in self.strategies:
            if current_time - rule.lastActive > GAagent.longtime:
                care_indices = [ind for ind in range(len(rule.condition)) if rule.condition[ind] != 2]
                if len(care_indices) > 1:
                    oneforth = round(len(care_indices) / 4)
                    if oneforth < 1:
                        oneforth = 1
                    chosen_oneforth = random.choices(care_indices, k=oneforth)

                    for c_ind in chosen_oneforth:
                        rule.condition[c_ind] = 2
                        rule.activeCount = 0
                        rule.lastActive = current_time
                        rule.actVar = GAagent.maxdev - (avgstrength + rule.specfactor)
                        if rule.actVar <= 0:
                            rule.actVar = rule.variance

                        rule.strength = avgstrength

                elif len(care_indices) == 1:
                    # if random.uniform(0, 1) < 1/2:
                    #     random.shuffle(rule.condition)
                    if 1 in rule.condition:
                        rule.condition[rule.condition.index(1)] = 0
                    else:
                        rule.condition[rule.condition.index(0)] = 1

                    rule.strength = avgstrength
                    rule.lastActive = current_time


class ProfitGA(GAagent):

    """
    possible values of fitness measure are 'profit_actVar', 'profit'.
    """
    fitness_measure = 'profit_actVar'

    def __init__(self, init_holding, initialcash, num_strategies, theta,
                 rule_keys, risk, ruleparam_range=None, ruleparam_center: Union[tuple, list, np.ndarray] = None):

        super().__init__(init_holding, initialcash, num_strategies, theta,
                         rule_keys, risk, ruleparam_range=ruleparam_range)

        StratRulesProfit.rule_fitness_measure = self.fitness_measure

        if ruleparam_range is None and ruleparam_center is None:
            self._maxpdcoeff = 1.2
            self._minpdcoeff = 0.7
            self._maxoffset = 19
            self._minoffset = -10

            self.strategies = [
                StratRulesProfit(variance=GAagent.init_var, gene_length=self.gene_length, profit=0)
                for _ in range(self.numStrategies)]
            self.defaultRule = StratRulesProfit(variance=GAagent.init_var,
                                                activecount=GAagent.mincount, profit=0,
                                                manual_condition=[2 for _ in range(self.gene_length)], tag='default')
        else:
            if ruleparam_center is None:
                self.pdcoef_center = 0.95
                self.offset_center = 4.5

                self._maxpdcoeff = self.pdcoef_center + ruleparam_range[0]
                self._minpdcoeff = self.pdcoef_center - ruleparam_range[0]
                self._maxoffset = self.offset_center + ruleparam_range[1]
                self._minoffset = self.offset_center - ruleparam_range[1]

            else:
                if ruleparam_center[0] is not None:
                    self.pdcoef_center = ruleparam_center[0]
                else:
                    self.pdcoef_center = 0.95

                if ruleparam_center[1] is not None:
                    self.offset_center = ruleparam_center[1]
                else:
                    self.offset_center = 4.5

                self._maxpdcoeff = self.pdcoef_center + ruleparam_range[0]
                self._minpdcoeff = self.pdcoef_center - ruleparam_range[0]
                self._maxoffset = self.offset_center + ruleparam_range[1]
                self._minoffset = self.offset_center - ruleparam_range[1]

            self.strategies = [
                StratRulesProfit(variance=GAagent.init_var, gene_length=self.gene_length,
                                 pdcoeff=random.uniform(self._minpdcoeff, self._maxpdcoeff),
                                 offset=random.uniform(self._minoffset, self._maxoffset), profit=0)
                for _ in range(self.numStrategies)]

            self.defaultRule = StratRulesProfit(variance=GAagent.init_var,
                                                activecount=GAagent.mincount, profit=0,
                                                pdcoeff=random.uniform(self._minpdcoeff, self._maxpdcoeff),
                                                offset=random.uniform(self._minoffset, self._maxoffset),
                                                manual_condition=[2 for _ in range(self.gene_length)], tag='default')

        self.bestForecast = None
        self.valid_actives = []
        self.previous_holding = self.holding

    def prepare_for_trading(self, ga_probability, mutation_per_bit, crossover_rate,
                            market, removal_replacement_ratio=0.5):

        if market.currentTime > 1 / ga_probability:
            # Using median or mean doesn't make qualitative difference
            self.perform_ga(ga_probability, mutation_per_bit, crossover_rate,
                            market, removal_replacement_ratio=removal_replacement_ratio)

        self.lForecast = self.forecast
        self.determine_activated(market)

        mincount = GAagent.mincount
        self.valid_actives = [active for active in self.activeList if active.activeCount >= mincount]
        nactive = len(self.valid_actives)

        try:
            # if market.currentTime < 5000:
            #     bestForecast = random.choice(valid_actives)
            # else:
            #     if random.uniform(0, 1) < 1/2:
            #         bestForecast = random.choice(sorted(valid_actives, key=attrgetter('strength'))[-4:])
            #     else:

            if self.fitness_measure == 'profit_actVar':
                self.bestForecast = max(self.valid_actives, key=lambda item: item.rule_profit - item.actVar)
            elif self.fitness_measure == 'profit':
                self.bestForecast = max(self.valid_actives, key=attrgetter('rule_profit'))

        except ValueError:
            self.bestForecast = None

        # Meaning some rule is active
        if nactive != 0:
            self.chosen_pdcoeff = self.bestForecast.pdcoeff
            self.chosen_offset = self.bestForecast.offset
            forecastvar = self.bestForecast.variance

        self.chosen_divisor = self.risk * forecastvar

    def helper_update_demand(self, p_trial, interest_rate):

        """
        Helper function to find other rule profits. We add this function because we do not want to
        change the actual demand, slope and forecast found by the original function where all agents
        used their best rules.

        :return: demand and slope of the agent's rule
        """

        forecast = self.chosen_pdcoeff * p_trial + self.chosen_offset

        if forecast > 0:
            demand = -((p_trial * (1 + interest_rate) - forecast) / self.chosen_divisor + self.holding)
            slope = (self.chosen_pdcoeff - 1 - interest_rate) / self.chosen_divisor
        else:
            demand = -(p_trial * (1 + interest_rate) / self.chosen_divisor + self.holding)
            slope = (- 1 - interest_rate) / self.chosen_divisor

        # Restrict trading volume
        if demand > self._maxdemand:
            demand = self._maxdemand
            slope = 0

        elif demand < -self._maxdemand:
            demand = -self._maxdemand
            slope = 0

        if demand > 0:
            if demand * p_trial > (self.cash - self._mincash):
                if self.cash - self._mincash > 0:
                    demand = (self.cash - self._mincash) / p_trial
                    slope = -demand / p_trial
                else:
                    demand = 0
                    slope = 0
                    if self.cash == self._mincash:
                        self.cash += 2000
        elif demand < 0.0 and demand + self.holding < self._minholding:
            demand = self._minholding - self.holding
            slope = 0.0

        return demand, slope

    def set_newrule_params(self, madv, medstrength):
        """
        set rule params based on strength, here profit
        :param madv:
        :param medstrength:
        :return:
        """
        if self.fitness_measure == 'profit':
            for rule in self._newrules:
                if rule.actVar < self.defaultRule.variance - madv:
                    rule.actVar = self.defaultRule.variance - madv
                    rule.strength = self.defaultRule.rule_profit - rule.specfactor

                if rule.actVar <= 0:
                    rule.actVar = np.median([rule.actVar for rule in self.all_rules])
                    rule.strength = medstrength

                rule.variance = rule.actVar

        elif self.fitness_measure == 'profit_actVar':
            for rule in self._newrules:

                if rule.actVar < self.defaultRule.variance - madv:
                    rule.actVar = self.defaultRule.variance - madv
                    rule.strength = self.defaultRule.rule_profit - rule.actVar - rule.specfactor
                    assert rule.actVar <= 100
                if rule.actVar <= 0:
                    rule.actVar = np.median([rule.actVar for rule in self.all_rules])
                    rule.strength = medstrength
                    assert rule.actVar <= 100

                rule.variance = rule.actVar

    def _add_newrules(self, offspring_strength, offspring_condition, offspring_pdcoeff,
                      offspring_offset, offspring_lastActive, offspring_activeCount=None):

        self._newrules.append(StratRulesProfit(strength=offspring_strength,
                                               manual_condition=offspring_condition,
                                               pdcoeff=offspring_pdcoeff,
                                               offset=offspring_offset,
                                               lastactive=offspring_lastActive,
                                               activecount=offspring_activeCount))

    def crossover(self, parent1, parent2, avgstrength, current_time):
        super().crossover(parent1, parent2, avgstrength, current_time)

        self._newrules[-1].actVar = (parent1.actVar + parent2.actVar) / 2
        self._newrules[-1].profit = (parent1.rule_profit + parent2.rule_profit) / 2
        assert self._newrules[-1].actVar <= 100


class TechnicalGA(ModifiedGA):

    def __init__(self, init_holding, initialcash, num_strategies, theta,
                 rule_keys, risk, force_mutationdiv=False, ruleparam_range=None,
                 ruleparam_center=None, modified_matrix=True):

        super().__init__(init_holding, initialcash, num_strategies, theta,
                         rule_keys, risk, ruleparam_range, ruleparam_center)

        self.force_mutationdiv = force_mutationdiv
        self.tag = 'tech_ga'
        # After Ehrentriech modified version
        self.F_tech = sum([rule.specfactor / 0.005 for rule in self.strategies]) / (num_strategies * self.gene_length)
        if modified_matrix is True:
            self.pmut = 1/3
        else:
            self.pmut = self.F_tech


    @property
    def new_pmut(self):
        return sum([rule.specfactor / 0.005 for rule in self.strategies]) / (self.numStrategies * self.gene_length)

    def mutate(self, selected_parent, mutation_per_bit, avgstrength, current_time, avg_var=None):

        if self.is_learning is True:
            offspring_condition = copy.deepcopy(selected_parent.condition)

            # Perform condition mutation. If slow store length functions in a variable
            # The weights preserve the average number of don't care (#) symbols, here
            # denoted as 2.
            changed = False
            bitchanged = False
            specificity = int(selected_parent.specfactor / 0.005)
            # Force diversity
            if self.force_mutationdiv:
                for i in range(len(offspring_condition)):
                    if random.uniform(0, 1) <= mutation_per_bit:

                        if offspring_condition[i] == 0:
                            offspring_condition[i] = random.choices((1, 2), weights=[self.pmut, 1 - self.pmut])[0]
                        bitchanged = changed = True
                        if offspring_condition[i] == 2:
                            specificity -= 1

                        elif offspring_condition[i] == 1:
                            offspring_condition[i] = random.choices((0, 2), weights=[self.pmut, 1 - self.pmut])[0]
                            bitchanged = changed = True
                            if offspring_condition[i] == 2:
                                specificity -= 1

                        else:
                            offspring_condition[i] = random.choices(
                                (0, 1, 2), weights=[1 / 2 * self.pmut, 1 / 2 * self.pmut, 1 - self.pmut])[0]
                            if offspring_condition[i] != selected_parent.condition[i]:
                                bitchanged = changed = True
                                specificity += 1

                if specificity == 0:
                    offspring_condition[random.randint(0, self.gene_length - 1)] = random.choice((0, 1))
                    bitchanged = changed = True

            # Don't force diversity
            else:
                for i in range(len(offspring_condition)):
                    if random.uniform(0, 1) <= mutation_per_bit:
                        if offspring_condition[i] == 0:
                            offspring_condition[i] = random.choices((1, 2), weights=[self.F_tech, 1 - self.F_tech])[0]
                            changed = bitchanged = True
                        elif offspring_condition[i] == 1:
                            offspring_condition[i] = random.choices((0, 2), weights=[self.F_tech, 1 - self.F_tech])[0]
                            changed = bitchanged = True
                        else:
                            offspring_condition[i] = random.choices(
                                (0, 1, 2), weights=[1 / 2 * self.F_tech, 1 / 2 * self.F_tech, 1 - self.F_tech])[0]
                            if offspring_condition[i] != selected_parent.condition[i]:
                                changed = bitchanged = True

            # Perform the forecast mutation
            r_pdcoeff = random.uniform(0, 1)
            if r_pdcoeff <= 0.2:
                changed = True
                offspring_pdcoeff = random.uniform(self._minpdcoeff, self._maxpdcoeff)

            elif 0.2 < r_pdcoeff <= 0.4:
                changed = True
                offspring_pdcoeff = random.uniform(
                    selected_parent.pdcoeff - 0.05 * (self._maxpdcoeff - self._minpdcoeff),
                    selected_parent.pdcoeff + 0.05 * (self._maxpdcoeff - self._minpdcoeff))
                if offspring_pdcoeff > self._maxpdcoeff:
                    offspring_pdcoeff = self._maxpdcoeff
                elif offspring_pdcoeff < self._minpdcoeff:
                    offspring_pdcoeff = self._minpdcoeff
            else:
                offspring_pdcoeff = selected_parent.pdcoeff

            r_offset = random.uniform(0, 1)
            if r_offset <= 0.2:
                changed = True
                offspring_offset = random.uniform(self._minoffset, self._maxoffset)

            elif 0.2 < r_offset <= 0.4:
                changed = True
                offspring_offset = random.uniform(
                    selected_parent.offset - 0.05 * (self._maxoffset - self._minoffset),
                    selected_parent.offset + 0.05 * (self._maxoffset - self._minoffset))
                if offspring_offset > self._maxoffset:
                    offspring_offset = self._maxoffset
                elif offspring_offset < self._minoffset:
                    offspring_offset = self._minoffset

            else:
                offspring_offset = selected_parent.offset

            offspring_strength = selected_parent.strength
            offspring_lastActive = current_time

            if changed:
                if selected_parent.activeCount == 0 or (current_time - selected_parent.lastActive) > GAagent.longtime:
                    offspring_strength = avgstrength
                    if not bitchanged:
                        offspring_lastActive = selected_parent.lastActive

                else:
                    if bitchanged:
                        offspring_strength = avgstrength
                    else:
                        offspring_lastActive = selected_parent.lastActive

            self._add_newrules(offspring_strength, offspring_condition, offspring_pdcoeff,
                               offspring_offset, offspring_lastActive)


class TechnicalGAdivMean(TechnicalGA):

    def __init__(self, init_holding, initialcash, num_strategies, theta,
                 rule_keys, risk, force_mutationdiv=False, ruleparam_range=None,
                 ruleparam_center=None):
        super().__init__(init_holding, initialcash, num_strategies, theta,
                         rule_keys, risk, force_mutationdiv=force_mutationdiv,
                         ruleparam_range=ruleparam_range, ruleparam_center=ruleparam_center)

    def update_demand(self, p_trial, dividend, interest_rate, *args, **kwargs):
        """

        :param interest_rate:
        :param dividend:
        :param p_trial: price set by the specialist
        :return: desired demand for the agent along with the demand derivative and a dummy variable
        which is used in calculating the trial price.
        """

        self.forecast = (p_trial + dividend) * self.chosen_pdcoeff + self.chosen_offset

        if self.forecast > 0:
            self.demand = -((p_trial * (1 + interest_rate) - self.forecast) / self.chosen_divisor + self.holding)
            self.slope = (self.chosen_pdcoeff - 1 - interest_rate) / self.chosen_divisor
        else:
            self.forecast = 0
            self.demand = -(p_trial * (1 + interest_rate) / self.chosen_divisor + self.holding)
            self.slope = (- 1 - interest_rate) / self.chosen_divisor

        # Restrict trading volume
        if self.demand > self._maxdemand:
            self.demand = self._maxdemand
            self.slope = 0

        elif self.demand < -self._maxdemand:
            self.demand = -self._maxdemand
            self.slope = 0

        self.constrain_demand(p_trial)

    def update_performance(self, market):
        d_bar = np.mean(market.divTimeSeries[-50:])
        a = 1 / self.theta
        b = 1 - a

        ftarget = market.price + d_bar

        for rule in self.activeList:
            rule.update_forecast(market.price, d_bar)

        if market.currentTime >= 1:
            for rule in self.oldActiveList:
                rule.update_actvar(ftarget, a, b)
                # assert rule.actVar <= 100


class TechnicalGAProfit(ProfitGA):
    _minholding = 0
    _mincash = 0

    def __init__(self, init_holding, initialcash, num_strategies, theta,
                 rule_keys, risk, force_mutationdiv=False, ruleparam_range=None, ruleparam_center=None):
        super().__init__(init_holding, initialcash, num_strategies, theta,
                         rule_keys, risk, ruleparam_range, ruleparam_center)
        self.force_mutationdiv = force_mutationdiv
        if self.gene_length != 32:
            raise NotImplementedError

        self.tag = 'tech_ga'
        # After Ehrentriech modified version
        self.F_tech = sum([rule.specfactor / 0.005 for rule in self.strategies]) / (num_strategies * self.gene_length)
        self._maxdemand = 1

    def update_fcast(self, p_trial, dividend):
        self.forecast = p_trial * self.chosen_pdcoeff + self.chosen_offset

    def update_performance(self, market):
        a = 1 / self.theta
        b = 1 - a

        ftarget = market.price

        for rule in self.activeList:
            rule.update_forecast_nodiv(market.price)

        if market.currentTime >= 1:
            for rule in self.oldActiveList:
                rule.update_actvar(ftarget, a, b)
                assert rule.actVar <= 100

    def perform_ga(self, probability, mutation_per_bit, crossover_rate,
                   market, removal_replacement_ratio):

        """

        :param probability:
        :param mutation_per_bit:
        :param crossover_rate:
        :return:
        """

        if random.uniform(0, 1) < probability:

            current_time = market.currentTime
            price = market.price

            avstrength = self.calculate_median()
            avg_var = np.median([rule.actVar for rule in self.all_rules])
            madv = self.calculate_madv()

            # Replace 20 worst rules at each invocation
            reject_list = sorted(self.strategies, key=attrgetter('strength'))[:self._nreject]

            while len(self._newrules) < self._nnew:
                parent1 = self.tournament()
                parent2 = None
                r = random.uniform(0, 1)

                if r < crossover_rate:
                    is_same_parent = True
                    while is_same_parent:
                        parent2 = self.tournament()
                        if parent1 != parent2:
                            is_same_parent = False

                    self.crossover(parent1, parent2, avstrength, current_time=current_time)
                else:
                    self.mutate(parent1, mutation_per_bit, avstrength, current_time=current_time, avg_var=avg_var)

            self.set_newrule_params(madv, avstrength)
            if random.uniform(0, 1) < removal_replacement_ratio:
                self.remove_worst()
                self.strategies += self._newrules
            else:
                self.replace_rules(reject_list)
            self._newrules = []
            self.generalize(avstrength, current_time)

    def mutate(self, selected_parent, mutation_per_bit, avgstrength, current_time, avg_var=None):

        if self.is_learning is True:
            offspring_condition = copy.deepcopy(selected_parent.condition)
            if avg_var is not None:
                avg_actVar = avg_var

            # Perform condition mutation. If slow store length functions in a variable
            # The weights preserve the average number of don't care (#) symbols, here
            # denoted as 2.
            changed = False
            bitchanged = False
            specificity = int(selected_parent.specfactor / 0.005)
            # Force diversity
            if self.force_mutationdiv:
                for i in range(len(offspring_condition)):
                    if random.uniform(0, 1) <= mutation_per_bit:

                        if offspring_condition[i] == 0:
                            offspring_condition[i] = random.choices((1, 2), weights=[self.F_tech, 1 - self.F_tech])[0]
                        bitchanged = changed = True
                        if offspring_condition[i] == 2:
                            specificity -= 1

                        elif offspring_condition[i] == 1:
                            offspring_condition[i] = random.choices((0, 2), weights=[self.F_tech, 1 - self.F_tech])[0]
                            bitchanged = changed = True
                            if offspring_condition[i] == 2:
                                specificity -= 1

                        else:
                            offspring_condition[i] = random.choices(
                                (0, 1, 2), weights=[1 / 2 * self.F_tech, 1 / 2 * self.F_tech, 1 - self.F_tech])[0]
                            if offspring_condition[i] != selected_parent.condition[i]:
                                bitchanged = changed = True
                                specificity += 1

                if specificity == 0:
                    offspring_condition[random.randint(0, self.gene_length - 1)] = random.choice((0, 1))
                    bitchanged = changed = True

            # Don't force diversity
            else:
                for i in range(len(offspring_condition)):
                    if random.uniform(0, 1) <= mutation_per_bit:
                        if offspring_condition[i] == 0:
                            offspring_condition[i] = random.choices((1, 2), weights=[self.F_tech, 1 - self.F_tech])[0]
                            changed = bitchanged = True
                        elif offspring_condition[i] == 1:
                            offspring_condition[i] = random.choices((0, 2), weights=[self.F_tech, 1 - self.F_tech])[0]
                            changed = bitchanged = True
                        else:
                            offspring_condition[i] = random.choices(
                                (0, 1, 2), weights=[1 / 2 * self.F_tech, 1 / 2 * self.F_tech, 1 - self.F_tech])[0]
                            if offspring_condition[i] != selected_parent.condition[i]:
                                changed = bitchanged = True

            # Perform the forecast mutation
            r_pdcoeff = random.uniform(0, 1)
            if r_pdcoeff <= 0.2:
                changed = True
                offspring_pdcoeff = random.uniform(self._minpdcoeff, self._maxpdcoeff)

            elif 0.2 < r_pdcoeff <= 0.4:
                changed = True
                offspring_pdcoeff = random.uniform(
                    selected_parent.pdcoeff - 0.05 * (self._maxpdcoeff - self._minpdcoeff),
                    selected_parent.pdcoeff + 0.05 * (self._maxpdcoeff - self._minpdcoeff))
                if offspring_pdcoeff > self._maxpdcoeff:
                    offspring_pdcoeff = self._maxpdcoeff
                elif offspring_pdcoeff < self._minpdcoeff:
                    offspring_pdcoeff = self._minpdcoeff

            else:
                offspring_pdcoeff = selected_parent.pdcoeff

            r_offset = random.uniform(0, 1)

            if r_offset <= 0.2:
                changed = True
                offspring_offset = random.uniform(self._minoffset, self._maxoffset)

            elif 0.2 < r_offset <= 0.4:
                changed = True
                offspring_offset = random.uniform(
                    selected_parent.offset - 0.05 * (self._maxoffset - self._minoffset),
                    selected_parent.offset + 0.05 * (self._maxoffset - self._minoffset))
                if offspring_offset > self._maxoffset:
                    offspring_offset = self._maxoffset
                elif offspring_offset < self._minoffset:
                    offspring_offset = self._minoffset

            else:
                offspring_offset = selected_parent.offset

            offspring_strength = selected_parent.strength
            offspring_lastActive = current_time
            offspring_actVar = selected_parent.actVar

            if changed:
                if selected_parent.activeCount == 0 or (current_time - selected_parent.lastActive) > GAagent.longtime:
                    offspring_strength = avgstrength
                    offspring_actVar = avg_actVar
                    if not bitchanged:
                        offspring_lastActive = selected_parent.lastActive

                else:
                    if bitchanged:
                        offspring_strength = avgstrength
                        offspring_actVar = avg_actVar
                    else:
                        offspring_lastActive = selected_parent.lastActive

            self._add_newrules(offspring_strength, offspring_condition, offspring_pdcoeff,
                               offspring_offset, offspring_lastActive)
            self._newrules[-1].actVar = offspring_actVar
            self._newrules[-1].rule_profit = selected_parent.rule_profit
            assert self._newrules[-1].actVar <= 100

    def generalize(self, avgstrength, current_time):
        avg_actVar = np.median([rule.actVar for rule in self.all_rules])

        for rule in self.strategies:
            if current_time - rule.lastActive > GAagent.longtime:
                care_indices = [ind for ind in range(len(rule.condition)) if rule.condition[ind] != 2]
                if len(care_indices) > 1:
                    oneforth = round(len(care_indices) / 4)
                    if oneforth < 1:
                        oneforth = 1
                    chosen_oneforth = random.choices(care_indices, k=oneforth)

                    for c_ind in chosen_oneforth:
                        rule.condition[c_ind] = 2

                    rule.activeCount = 0
                    rule.lastActive = current_time
                    rule.actVar = avg_actVar

                    if rule.actVar <= 0:
                        rule.actVar = rule.variance
                    rule.strength = avgstrength
                    rule.rule_profit = 0

                elif len(care_indices) == 1:
                    # if random.uniform(0, 1) < 1/2:
                    #     random.shuffle(rule.condition)
                    if 1 in rule.condition:
                        rule.condition[rule.condition.index(1)] = 0
                    else:
                        rule.condition[rule.condition.index(0)] = 1

                    rule.strength = avgstrength
                    rule.lastActive = current_time
                assert rule.actVar <= 100


class NoiseTrader(Agent):
    _maxdemand = 1

    minpdcoeff = 0.7
    maxpdcoeff = 1.2
    minoffset = -10
    maxoffset = 19

    def __init__(self, init_holding, initialcash, risk, theta, variance=None):
        super().__init__(init_holding, initialcash, risk)
        self.pdcoeff = random.uniform(self.minpdcoeff, self.maxpdcoeff)
        self.offset = random.uniform(self.minoffset, self.maxoffset)
        self.tag = 'noise'
        if variance is None:
            self.variance = 4
        else:
            self.variance = variance
        self.theta = theta

    def update_demand(self, p_trial, dividend, interest_rate, *args, **kwargs):
        expected_forecast = self.pdcoeff * (p_trial + dividend) + self.offset

        if expected_forecast > 0:
            desired_demand = (expected_forecast - p_trial * (1 + interest_rate)) / (self.risk * self.variance)
            self.demand = desired_demand - self.holding
            self.slope = (self.pdcoeff - 1 - interest_rate) / (self.risk * self.variance)
        else:
            desired_demand = -p_trial * (1 + interest_rate) / (self.risk * self.variance)
            self.demand = desired_demand - self.holding
            self.slope = - 1 - interest_rate / (self.risk * self.variance)

        if self.demand > self._maxdemand:
            self.demand = self._maxdemand
        if self.demand < - self._maxdemand:
            self.demand = - self._maxdemand

        self.constrain_demand(p_trial)

    def update_variance(self, price, dividend):
        # TODO write the variance for noise trader
        self.variance = None

    def randomize_parameters(self):
        self.pdcoeff = random.uniform(self.minpdcoeff, self.maxpdcoeff)
        self.offset = random.uniform(self.minoffset, self.maxoffset)


class NoiseHerder(NoiseTrader):
    def __init__(self, init_holding, initialcash, risk, herding_factor, variance=None):
        super().__init__(init_holding, initialcash, risk, variance=variance)
        self.alpha = herding_factor
        self.tag = 'herder'

    def update_demand(self, p_trial, dividend, interest_rate, measure_other_agents=None, *args, **kwargs):
        discrepancy = measure_other_agents - 1
        expected_forecast = (self.pdcoeff + self.alpha * discrepancy) * (p_trial + dividend) + self.offset

        if expected_forecast > 0:
            desired_demand = (expected_forecast - p_trial * (1 + interest_rate)) / (self.risk * self.variance)
            self.demand = desired_demand - self.holding
            self.slope = (self.pdcoeff - 1 - interest_rate) / (self.risk * self.variance)
        else:
            desired_demand = -p_trial * (1 + interest_rate) / (self.risk * self.variance)
            self.demand = desired_demand - self.holding
            self.slope = - 1 - interest_rate / (self.risk * self.variance)

        if self.demand > self._maxdemand:
            self.demand = self._maxdemand
        if self.demand < - self._maxdemand:
            self.demand = - self._maxdemand

        self.constrain_demand(p_trial)


def ewma(series, history, n):
    alpha = 2 / (n + 1)
    ma = [(series[-history] + sum([series[-history - t] * (1 - alpha) ** t for t in range(1, n)])) /
          sum([(1 - alpha) ** i for i in range(n)])]
    for i in reversed(range(1, history)):
        ma.append(alpha * series[-i] + (1 - alpha) * ma[-1])
    return ma


# noinspection PyIncorrectDocstring
class TrendRegressor(Agent):
    minvar = 1e-6
    _mincash = 0
    _minholding = 0

    def __init__(self, init_holding, initialcash, risk, tau, gamma, trendtype='linreg',
                 variance=None, maxvar=None, maxdemand=None):
        """

        :param tau: time horizon
        :param variance:
        :param gamma: extrapolation rate
        """
        super().__init__(init_holding, initialcash, risk)
        if variance is None:
            self.variance = 4
        else:
            self.variance = variance
        self.tag = 'trend' + str(int(tau))
        self.trendtype = trendtype

        if tau > 500:
            raise ValueError("Maximum tau value is 500")
        self.tau = tau

        self.gamma = gamma
        self.pdcoeff = 0
        self.offset = 0

        if maxvar is None:
            self.maxvar = None
        else:
            self.maxvar = maxvar

        if maxdemand is None:
            self.maxdemand = 2
        else:
            self.maxdemand = maxdemand

    # TODO The trend functions look only into the asset prices and hence are technical analysis
    # however I wonder if looking into price cum dividend series (through RNNS for example)
    # is equivalent to technical and fundamental analysis combined?

    def update_variance(self, price_series, dividend_series):
        self.variance = (1 - 1 / self.tau) * self.variance + 1 / self.tau * ((price_series[-1] + dividend_series[-1]) -
                        (self.pdcoeff * (price_series[-2] + dividend_series[-2]) + self.offset)) ** 2

    def svr_trend(self, independent_var):
        if self.forecast != 0:
            self.variance = (1 - 1 / self.tau) * self.variance + 1 / self.tau * (
                    (independent_var[-1] - self.forecast) ** 2)

        y = independent_var[-self.tau:]
        X = np.arange(self.tau).reshape(-1, 1)
        regr = make_pipeline(StandardScaler(), SVR(C=200, epsilon=0.1))
        regr.fit(X, y)
        self.forecast = regr.predict(np.array(self.tau + 1).reshape(1, -1))[0]

    def linreg_trend(self, price_series, dividend_series, independent_var):
        """

        :param independent_var: independent variable for the regression.
        :return: linear regression
        """

        reg = linear_model.LinearRegression().fit(independent_var,
                                                  price_series[-self.tau:])

        if self.pdcoeff != 0 or self.offset != 0:
            self.update_variance(price_series, dividend_series)

        if self.variance < TrendRegressor.minvar:
            self.variance = TrendRegressor.minvar

        if self.maxvar is not None:
            if self.variance > self.maxvar:
                self.variance = self.maxvar

        self.pdcoeff = reg.coef_[0]
        self.offset = reg.intercept_

    def momentum_trend(self, horizon, independent_var):
        self.pdcoeff = (independent_var[-1] - independent_var[-horizon]) / horizon
        self.offset = independent_var[-1]

    def expodecaying_trend(self, price_series, dividend_series, independent_var):
        """
        :return: exponentially weighted moving average
        """

        # history goes back in my code
        exp_ma = ewma(independent_var, self.tau+1, 9)
        reg = linear_model.LinearRegression().fit(np.array(exp_ma[-self.tau - 1: -1]).reshape(-1, 1),
                                                  exp_ma[-self.tau:])

        if self.pdcoeff != 0 or self.offset != 0:
            self.update_variance(price_series, dividend_series)

        if self.variance < TrendRegressor.minvar:
            self.variance = TrendRegressor.minvar

        if self.maxvar is not None:
            if self.variance > self.maxvar:
                self.variance = self.maxvar

        self.pdcoeff = reg.coef_[0]
        self.offset = reg.intercept_

    def update_trendfcast(self, price_series, dividend_series, independent_var):
        if self.trendtype == 'linreg':
            self.linreg_trend(price_series, dividend_series, independent_var)
            self.forecast = self.pdcoeff * (price_series[-1] + dividend_series[-1]) + self.offset

        # elif self.trendtype == 'svr':
        #     self.svr_trend(independent_variable)
        #
        elif self.trendtype == 'expodecay':
            self.expodecaying_trend(price_series, dividend_series, independent_var)
            self.forecast = self.pdcoeff * (price_series[-1] + dividend_series[-1]) + self.offset

    def update_demand(self, p_trial, interest_rate, dividend=None, *args, **kwargs):

        divisor = self.variance * self.risk

        if self.trendtype == 'expodecay':

            if self.forecast > 0:
                self.demand = -(self.gamma * (p_trial * (1 + interest_rate) - self.forecast) / divisor + self.holding)
                self.slope = - self.gamma * (1 + interest_rate) / divisor
            else:
                self.forecast = 0
                self.demand = -(self.gamma * p_trial * (1 + interest_rate) / divisor + self.holding)
                self.slope = - self.gamma * (1 + interest_rate) / divisor

        elif self.trendtype == 'linreg' or self.trendtype == 'svr':

            if self.forecast > 0:
                self.demand = -((p_trial * (1 + interest_rate) - self.forecast) / divisor + self.holding)
                self.slope = (self.pdcoeff - 1 - interest_rate) / divisor
            else:
                self.forecast = 0
                self.demand = -(p_trial * (1 + interest_rate) / divisor + self.holding)
                self.slope = (- 1 - interest_rate) / divisor

        if self.demand > self.maxdemand:
            self.demand = self.maxdemand
        if self.demand < -self.maxdemand:
            self.demand = -self.maxdemand

        self.constrain_demand(p_trial)


class ContrarianTraders:
    def __init__(self):
        pass


class Imitators(Agent):
    maxdemand = 5

    # Do I want my imitators to trade only based on others or opt to do noise trading as well?
    def __init__(self, init_holding, initialcash, risk):
        super().__init__(init_holding, initialcash, risk)

    # This demand is based on imitation and only from GA agents.
    def demand(self, presented_rules, p_trial, dividend, interest_rate):
        # First the agent must choose a rule to imitate
        normalizedweights = np.divide([pr.strength for pr in presented_rules],
                                      sum([pr.strength for pr in presented_rules]))
        selectedrule = random.choices(presented_rules, weights=normalizedweights)[0]

        expected_forecast = selectedrule.forecast(p_trial, dividend)
        desired_demand = (expected_forecast - p_trial * (1 + interest_rate)) / \
                         (self.risk * selectedrule.forecast_accuracy)
        effective_demand = desired_demand - self.holding
        demand_derivative = (selectedrule.pdcoeff - 1 - interest_rate) / \
                            (self.risk * selectedrule.forecast_accuracy)

        if self.demand > self.maxdemand:
            self.demand = self.maxdemand
        if self.demand < -self.maxdemand:
            self.demand = -self.maxdemand

        return effective_demand, demand_derivative


class NetworkedAgents:

    def __init__(self, competent_agents, imitators, imtocomp_probability, imtoim_probability):
        """

        :param competent_agents:
        :param imitators:
        :param imtocomp_probability:
        :param imtoim_probability:
        """
        # Competent agents are GAagents or reinforcement learning agents
        # or any other algorithmic trading agent
        competent_keys = ['comp' + str(i) for i in range(len(competent_agents))]
        imitator_keys = ['imit' + str(i) for i in range(len(imitators))]
        self.competent_agents = {key: agent for key, agent in zip(competent_keys, competent_agents)}
        self.imitators = {key: agent for key, agent in zip(imitator_keys, imitators)}

        G = nx.Graph()
        G.add_nodes_from(competent_keys + imitator_keys)
        # Build the edges by code
        # First establish the edges from imitators to competents
        # mean followers are determined by their strength times number of imitators divided by 5
        # This mean becomes the mean of a standard normal distribution
        strengths = [comp.average_strength() for comp in competent_agents]
        normalized_strengths = np.divide(strengths, sum(strengths))
        mean_followers = np.around(np.multiply(normalized_strengths, len(imitators)) * imtocomp_probability)
        ncompetent_followers = [round(random.normalvariate(mu, 1)) for mu in mean_followers]
        for i in range(len(ncompetent_followers)):
            if ncompetent_followers[i] > len(imitators):
                ncompetent_followers[i] = len(imitators)
            if ncompetent_followers[i] < 0:
                ncompetent_followers[i] = 0
        edge_list = []
        imitator_bin = np.arange(len(imitators))
        for n in range(len(competent_agents)):
            followers = np.random.choice(imitator_bin, size=ncompetent_followers[n])
            this_edge_list = [('comp' + str(n), 'imit' + str(im)) for im in followers]
            edge_list += this_edge_list
            n += 1

        # make edges from imitators to each other
        for i in range(len(imitators)):
            this_edge_list = [('imit' + str(i), 'imit' + str(im)) for im in range(len(imitators))
                              if random.uniform(0, 1) < imtoim_probability]
            edge_list += this_edge_list
        G.add_edges_from(edge_list)
        self.network = G

    def draw_network(self):
        nx.draw_networkx(self.network)
