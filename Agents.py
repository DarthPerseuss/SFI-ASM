from Stocks import Stock
import random
import copy
from operator import attrgetter
from statistics import median, mean
import numpy as np


class StrategyRules:

    def __init__(self, gene_length=None, manual_condition=None, parameter_a=None,
                 parameter_b=None, forecast_accuracy=None):
        """
        
        :param gene_length: number of conditions
        :param manual_condition: Used to write custom conditions for example
        after mutation and crossover
        :param parameter_a: parameter a in the forecast
        :param parameter_b: parameter b in the forecast
        :param forecast_accuracy: Used to set the accuracy of the rule manually
        if not None
        """

        # The condition strings are initiated such that they get # (don't care) symbols
        # with a probability of 0.8 and the other two with probabilities of 0.1 each.
        # Here 0 means false 1 means True and 2 means "dont'care".
        if gene_length is not None and manual_condition is None:
            self.condition = random.choices([0, 1, 2], weights=[1, 1, 8], k=gene_length)

        if gene_length is not None and manual_condition is not None:
            raise ValueError("Gene length and manual condition can't be both set")

        if type(manual_condition) == list and gene_length is None:
            self.condition = manual_condition

        if type(manual_condition) != list and gene_length is None:
            raise TypeError("The agent's condition string should be represented as a list")

        # The paramaters a and b are initiated here, but changed in the Agent class
        # since they undergo crossovers. They are defined as properties.
        if parameter_a is not None:
            self.parameter_a = parameter_a
        else:
            self.parameter_a = random.uniform(0.7, 1.2)
        if parameter_b is not None:
            self.parameter_b = parameter_b
        else:
            self.parameter_b = random.uniform(-10, 19)
        # TODO: check initial forecast accuracy
        if forecast_accuracy is None:
            self.forecast_accuracy = [10.]
        else:
            self.forecast_accuracy = [forecast_accuracy]
        # TODO: check the initial forecast error
        self.forecastError = 10.
        self.nonactivatedPeriodCount = 0
        # 10 is set as the maximum forecast error change it if you must.
        self._maxError = 10

    @property
    def specificity(self):
        """
        The specificity parameter is computed after each change in the rules.
        Based on the addition term given in the init method.
        :return: Number of specific symbols (|0| + |1|)
        """
        return self.condition.count(0) + self.condition.count(1)

    @property
    def fitness(self):
        return 100 - (self.forecast_accuracy[-1] + 0.15 * self.specificity)

    @fitness.setter
    def fitness(self, value):
        self.fitness = value

    def forecast(self, price, dividend):
        """

        :param price: this period's market price
        :param dividend: this period's market dividend
        :return: Expected returns for next period
        """
        return self.parameter_a * (price + dividend) + self.parameter_b

    # The function below should be applied one time-step later than the actual forecast accuracy is made
    # We want to compare the actual price to the forecast the agents made based on the previous time step.
    # So the current price and dividend come from the market and the previous price and dividend determine
    # the agent's forecast accuracy.
    def update_forecastaccuracy(self, current_price, current_dividend, previous_price, previous_dividend, theta):
        v_next = (1 - 1 / theta) * self.forecast_accuracy[-1] + (1 / theta) * ((current_price + current_dividend) - (
                self.parameter_a * (previous_price + previous_dividend) + self.parameter_b)) ** 2
        self.forecast_accuracy.append(v_next)

    def calculate_forecasterror(self, price_timeseries, dividend_timeseries):
        self.forecastError = price_timeseries[-1] + dividend_timeseries[-1] - \
                             (self.parameter_a * (price_timeseries[-2] + dividend_timeseries[-2]) + self.parameter_b)
        # Error is bounded
        if self.forecastError > self._maxError:
            self.forecastError = self._maxError

    def generalize_condition(self,  median_fitness):
        """
        Method that converts 1/4 of True or False symbols to
        don't care (#) symbols inside the condition part.
        :param median_fitness: median fitness in the agent's rule set to set
        the fitness of a generalized rule in case it has undergone generalization
        :return: generalized rule
        """
        if self.nonactivatedPeriodCount > 4000:
            care_indices = [ind for ind in range(len(self.condition)) if self.condition[ind] != 2]
            oneforth = round(len(care_indices))
            chosen_oneforth = random.choices(care_indices, k=oneforth)
            for c_ind in chosen_oneforth:
                self.condition[c_ind] = 2
            self.fitness = median_fitness

    def is_activated(self, market_condition):
        """
        Checks whether the market state and the agent's condition match.
        Should be run at each timestep.
        :param market_condition: Taken from the actual market dynamics
        :return: Truth value
        """
        is_matched = []
        for c in range(len(market_condition)):
            if self.condition[c] == 2 or market_condition[c] == 2:
                is_matched.append(True)
            elif self.condition[c] == market_condition[c]:
                is_matched.append(True)
            else:
                is_matched.append(False)
        if set(is_matched) == {True}:
            self.nonactivatedPeriodCount = 0
            return True
        else:
            self.nonactivatedPeriodCount += 1
            return False


class Agent(Stock):

    def __init__(self, init_holding, initialcash, num_strategies, theta,
                 gene_length, risk_coef, num_shares, init_price,
                 dividend_startvalue, rho, noise_sd, d_bar, interest_rate):
        super().__init__(num_shares, init_price, dividend_startvalue, rho, noise_sd, d_bar, interest_rate)
        self.holdingsSeries = [init_holding]
        self.cash = initialcash
        self.holding = init_holding
        self.interestRate = interest_rate
        self.wealth = [initialcash]
        self.strategyLength = gene_length
        self.numStrategies = num_strategies
        self.strategies = [StrategyRules(gene_length=gene_length) for _ in range(num_strategies)]
        # theta measures how much the agents look into the past
        self.theta = theta
        self.risk_coef = risk_coef
        self._offsprings = []

    def set_holding(self, value):
        self.holding += value
        if self.holding < 0:
            self.holding = 0
        self.holdingsSeries.append(self.holding)

    def set_cash(self, value):
        self.cash += value
        if self.cash < 0:
            self.cash = 0

    def select_demandrule(self):
        """
        Method to determine the rule that will be selected to compute the demand
        :return: selected rule j
        """
        activated_rules = []
        # random.sample() shuffles not in place, we want to shuffle to get
        # some randomness with regards to the chosen rules when their forecast
        # accuracies are equal. This happens more often in the first periods
        for rule in random.sample(self.strategies, self.numStrategies):
            if rule.is_activated(market_condition=self.marketCondition):
                activated_rules.append(rule)
        # Select the rule with maximum forecast_accuracy
        try:
            selected_rule = max(activated_rules, key=attrgetter('forecast_accuracy'))
            return selected_rule
        # if no rule is activated return None. Since empty sequence
        # leads to value error with max().
        except ValueError or KeyError:
            return None

    def demand(self, p_trial, selected_rule):
        """

        :param p_trial: price set by the specialist
        :param selected_rule: rule with the highest forecast accuracy, type: StrategyRules
        :return: desired demand for the agent along with the demand derivative and a dummy variable
        which is used in calculating the trial price.
        """
        # TODO: is this all that needs to be implemented?
        # if no rules matched the market condition use the weighted fitness average
        # of parameters b and a and the average forecast accuracy of last period. 
        if selected_rule is None:

            rules_fitness = [self.strategies[i].fitness for i in range(self.numStrategies)]
            average_a = np.dot(rules_fitness, [self.strategies[i].parameter_a for i in range(self.numStrategies)]) \
                        / sum(rules_fitness)
            average_b = np.dot(rules_fitness, [self.strategies[j].parameter_b for j in range(self.numStrategies)]) \
                        / sum(rules_fitness)
            expected_forecast = average_a * (p_trial + self.divTimeSeries[-1]) + average_b
            expected_forecast_accuracy = mean([self.strategies[i].forecast_accuracy[-1]
                                               for i in range(self.numStrategies)])
            # effective demand
            desired_demand = (expected_forecast - p_trial * (1+self.interestRate)) / \
                             (self.risk_coef * expected_forecast_accuracy)
            effective_demand = desired_demand - self.holding
            # demand derivative
            demand_derivative = (average_a - 1 - self.interestRate) / \
                                (self.risk_coef * selected_rule.forecast_accuracy[-1])

        else:
            expected_forecast = selected_rule.forecast(p_trial, self.divTimeSeries[-1])
            desired_demand = (expected_forecast - p_trial * (1+self.interestRate)) / \
                             (self.risk_coef * selected_rule.forecast_accuracy[-1])
            effective_demand = desired_demand - self.holding
            demand_derivative = (selected_rule.parameter_a - 1 - self.interestRate) / \
                                (self.risk_coef * selected_rule.forecast_accuracy[-1])

        return effective_demand, demand_derivative

    def mutate(self, mutation_per_period, mutation_per_bit):
        """
        The mutation of the forecasts is done in 3 ways as well. With probability 0.2,
        they are uniformly changed to a value within the permissible ranges of the parameters
        which is[0.7, 1.2] for the a parameter and[ −10.0, 19.0] for the b parameter. With
        probability 0.2, the current parameter value is uniformly distributed within ± 5%
        of its current value, and it is left unchanged for the remaining cases.
        
        :param mutation_per_period: 
        :param mutation_per_bit: 
        :return: mutated offspring
        """
        if random.uniform(0, 1) < mutation_per_period:
            fitness_array = [self.strategies[i].fitness for i in range(self.numStrategies)]
            selected_parent = random.choice(self.strategies, weights=fitness_array)
            offspring_condition = copy.deepcopy(selected_parent.condition)
            offspring_forecast_a = selected_parent.parameter_a 
            offspring_forecast_b = selected_parent.parameter_b

            # Perform condition mutation. If slow store length functions in a variable
            # The weights preserve the average number of don't care (#) symbols, here
            # denoted as 2.
            for i in range(len(offspring_condition)):
                if random.uniform(0, 1) <= mutation_per_bit:
                    if offspring_condition == 0:
                        offspring_condition = random.choice((1, 2), weights=[1, 2])
                    elif offspring_condition == 1:
                        offspring_condition = random.choice((0, 2), weights=[1, 2])
                    else:
                        offspring_condition = random.choice((0, 1, 2))
            
            # Perform the forecast mutation 
            r_forecast = random.uniform(0, 1)
            if r_forecast <= 0.2:
                offspring_forecast_a = random.uniform(0.7, 1.2)
                offspring_forecast_b = random.uniform(-10, 19)
            elif 0.2 < r_forecast <= 0.4:
                offspring_forecast_a = random.uniform(offspring_forecast_a - offspring_forecast_a*0.05, 
                                                       offspring_forecast_a + offspring_forecast_a*0.05)
                offspring_forecast_b = random.uniform(offspring_forecast_b - offspring_forecast_b * 0.05,
                                                       offspring_forecast_b + offspring_forecast_b * 0.05)
            else:
                pass
            # Calculate offspring forecast accuracy as the median forecast error over all rules
            # in the agent's set of rules
            offspring_forecastaccuracy = median([self.strategies[i].calculate_forecasterror(self.priceTimeSeries,
                                                 self.divTimeSeries) for i in range(self.numStrategies)])
            assert type(offspring_forecastaccuracy) == float

            self._offsprings.append(StrategyRules(manual_condition=offspring_condition,
                                                  parameter_a=offspring_forecast_a,
                                                  parameter_b=offspring_forecast_b,
                                                  forecast_accuracy=offspring_forecastaccuracy))

    def crossover(self, crossover_rate):
        """
        Chooses two parents by tournament selection and performs uniform crossover between
        them to get the new conditions. Performs one of three tasks to perform the crossover
        for the forecasts with equal probability. One is to choose a parent and get both genes
        from that single parent. The other is to randomly choose each of the parents for each
        gene. The last method uses a weighted average of both parents to determine a and b
        parameters in the offspring.

        :param crossover_rate: rate at which crossover occurs
        :return: crossovered offspring
        """
        parent1 = None
        parent2 = None
        if random.uniform(0, 1) <= crossover_rate:
            is_same_parent = True
            while is_same_parent:
                parent1_candidates = random.choices(self.strategies, k=2)
                parent1 = random.choice(parent1_candidates, weights=[parent1_candidates[0].fitness,
                                                                     parent1_candidates[1].fitness])

                parent2_candidates = random.choices(self.strategies, k=2)
                parent2 = random.choice(parent1_candidates, weights=[parent2_candidates[0].fitness,
                                                                     parent2_candidates[1].fitness])
                if parent1 != parent2:
                    is_same_parent = False
            # This line performs the uniform corssover
            offspring_condition = [random.choice(parent1.condition[i], parent2.condition[i])
                                   for i in range(self.strategyLength)]

            # This random number determines which strategy is used for the parameters of the offspring
            r = random.uniform(0, 1)
            if r < 1 / 3:
                chosen_parent = random.choice((parent1, parent2))
                crossed_parameter_a = chosen_parent.parameter_a
                crossed_parameter_b = chosen_parent.parameter_b
            elif 1 / 3 <= r < 2 / 3:
                chosenparent_for_param_a = random.choice((parent1, parent2))
                chosenparent_for_param_b = random.choice((parent1, parent2))
                crossed_parameter_a = chosenparent_for_param_a.parameter_a
                crossed_parameter_b = chosenparent_for_param_b.parameter_b
            else:
                # The forecast accuracy is a list but only the immediate entry is relevant.
                weights = [1 / parent1.forecast_acccuracy[-1], 1 / parent2.forecast_acccuracy[-1]]
                normalized_weights = [1 / sum(weights) * w for w in weights]
                crossed_parameter_a = normalized_weights[0] * parent1.parameter_a + \
                                     normalized_weights[1] * parent2.parameter_a
                crossed_parameter_b = normalized_weights[0] * parent1.parameter_b + \
                                     normalized_weights[1] * parent2.parameter_b

            offspring_forecastaccuracy = (parent1.forecast_accuracy + parent2.forecast_accuracy) / 2

            self._offsprings.append(StrategyRules(manual_condition=offspring_condition,
                                                  parameter_a=crossed_parameter_a,
                                                  parameter_b=crossed_parameter_b,
                                                  forecast_accuracy=offspring_forecastaccuracy))

    def select(self, final=False):
        """
        Remove the worst 20 rules and replace them by the offsprings
        :param final: Checks if the rule set exceeds the number of strategies
        each agent is defined to have and removes the extra worst ones. Should
        be done at the last step of the simulation
        :return: new strategy rule set
        """
        if len(self.strategies) >= self.numStrategies and len(self._offsprings) >= 20:
            self.strategies = self.strategies.sort(key=attrgetter('forecast_accuracy'))
            del self.strategies[0:19]
            self.strategies += self._offsprings
            # empty the offspring list after concatenation
            self._offsprings = []

        if final is True:
            if len(self.strategies) > self.numStrategies:
                extra = len(self.strategies) - self.numStrategies
                del self.strategies.sort(key=attrgetter('forecast_accuracy'))[0: extra]
