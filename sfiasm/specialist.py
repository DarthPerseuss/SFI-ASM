# import warnings
import math
import numpy as np


class Specialist:

    def __init__(self, max_stockprice, min_stockprice, min_excess, eta, max_iterations,
                 theta, taup):

        """
        :param max_stockprice:
        :param min_stockprice:
        :param min_excess: excess demand needs to be smaller than this number if the price setting process
        is to stop
        :param eta:
        :param max_iterations:
        """
        self.maxprice = max_stockprice
        self.minprice = min_stockprice
        self.eta = eta
        self.minExcess = min_excess
        self.maxIterations = max_iterations
        # Theta is an agent property but for convenience in this case we have made
        # it part of the specialist.
        self.theta = theta
        self.volume = 0
        self.offerfrac = 0
        self.bidfrac = 0
        self.mincount = 5
        self.rea = 6.333
        self.reb = 16.6882
        self.taupnew = -math.expm1(-1/taup)
        self.taupdecay = 1 - self.taupnew

    def determine_price(self, agents, market, mode='default', herder=False):

        # _______Obj_C Implementation_______
        done = False
        scount = 0
        trialprice = 0

        bidtotal = 0
        offertotal = 0
        slopetotal = 0

        if herder is True:
            some_measure_agents = np.mean([agent.chosen_pdcoeff if 'ga' in agent.tag
                                           else agent.pdcoeff for agent in agents])

        # dividend_mean = np.mean(market.divTimeSeries[-50:])

        # if self.trend_independent_var is not None:
        #     if self.trend_independent_var == 'price':
        #         independent_var = np.add(market.priceTimeSeries[-500:], 10)
        #     elif self.trend_independent_var == 'pricecumdiv':
        #         independent_var = np.add(market.priceTimeSeries[-500:], market.divTimeSeries[-500:])
        #     else:
        #         raise ValueError('The only accepted values are "price" and "pricecumdiv".')

        while scount < self.maxIterations and not done:

            if mode == 'ree':
                trialprice = self.rea * market.dividend + self.reb
                break

            elif mode == 'default':
                if scount == 0:
                    trialprice = market.price

                else:
                    imbalance = bidtotal - offertotal
                    if abs(imbalance) <= self.minExcess:
                        done = True
                        continue

                    if slopetotal != 0:
                        trialprice -= imbalance / slopetotal
                    else:
                        trialprice *= 1 + self.eta * imbalance

            bidtotal = 0
            offertotal = 0
            slopetotal = 0

            for agent in agents:

                if 'trend' in agent.tag:
                    if agent.trendtype == 'linreg':
                        agent.update_trendfcast(market.priceTimeSeries[-agent.tau:],
                                                market.divTimeSeries[-agent.tau:],
                                                np.array(market.priceTimeSeries[agent.tau-1: -1]).reshape(-1, 1))
                    elif agent.trendtype == 'expodecay':
                        agent.update_trendfcast(market.priceTimeSeries[-agent.tau:],
                                                market.divTimeSeries[-agent.tau:],
                                                np.array(market.priceTimeSeries[-2 * (agent.tau+9):]))
                    agent.update_demand(trialprice, market.r)

                elif agent.tag == 'herder':
                    agent.update_demand(trialprice, market.dividend, market.r, measure_other_agents=some_measure_agents)

                elif 'tech' in agent.tag:
                    agent.update_demand(trialprice, market.dividend, market.r)

                else:
                    agent.update_demand(trialprice, market.dividend, market.r)

                slopetotal += agent.slope
                if agent.demand > 0:
                    bidtotal += agent.demand
                elif agent.demand < 0:
                    offertotal -= agent.demand

            if trialprice > self.maxprice:
                trialprice = self.maxprice
            if trialprice < self.minprice:
                trialprice = self.minprice

            if bidtotal > offertotal:
                self.volume = offertotal
            else:
                self.volume = bidtotal

            if bidtotal > 0.:
                self.bidfrac = self.volume / bidtotal
            else:
                self.bidfrac = 0.

            if offertotal > 0.:
                self.offerfrac = self.volume / offertotal
            else:
                self.offerfrac = 0.

            scount += 1
            # if scount == 10:
            #     print('Market is rationed!')
            #     print(sum(agent.demand for agent in agents))

        market.set_price(trialprice)

    def clear_market(self, agents, market, mode=None):

        if mode is None:
            self.determine_price(agents, market, mode='default')
        else:
            self.determine_price(agents, market, mode=mode)

        bfp = self.bidfrac * market.price
        ofp = self.offerfrac * market.price
        tp = self.taupnew * market.profitperunit

        for agent in agents:
            agent.profit = self.taupdecay * agent.profit + tp * agent.holding

            if agent.demand > 0:
                agent.adjust_holding(agent.demand * self.bidfrac)
                agent.adjust_cash(-agent.demand * bfp)

            else:
                agent.adjust_holding(agent.demand * self.offerfrac)
                agent.adjust_cash(-agent.demand * ofp)

    def determine_profit_and_update_performances(self, agents, market):
        """
        For now this only works for TechnicalGA(ProfitGA) agents
        :param agents:
        :param market:
        :return:
        """
        if str(type(agents[0])) != "<class 'agents.TechnicalGA'>":
            raise TypeError(f'agents are of type {type(agents[0])} whereas type '
                            f'<class agents.TechnicalGA > was expected')

        for i in range(len(agents)):
            other_agents = [agents[ind] for ind in range(len(agents)) if ind != i]

            if len(agents[i].valid_actives) <= 1:
                continue

            done = False
            scount = 0
            trialprice = 0

            bidtotal = 0
            offertotal = 0
            slopetotal = 0

            for rule in [valid for valid in agents[i].valid_actives if valid != agents[i].bestForecast]:
                rule_demand = 0
                bidfrac = 0
                offerfrac = 0
                rule.previous_clearing_price = rule.current_clearing_price

                while scount < self.maxIterations and not done:

                    if scount == 0:
                        # Since this function will be run after the market price is set in the current iteration
                        # we want to set the price two periods ago as the trial price.
                        trialprice = market.priceTimeSeries[-2]

                    else:
                        imbalance = bidtotal - offertotal
                        # Imbalance will have an effect on speed and accuracy,
                        # needs to be fine-tuned.
                        if abs(imbalance) <= 0.1:
                            done = True
                            continue

                        if slopetotal != 0:
                            trialprice -= imbalance / slopetotal
                        else:
                            trialprice *= 1 + self.eta * imbalance

                    if trialprice > self.maxprice:
                        trialprice = self.maxprice
                    if trialprice < self.minprice:
                        trialprice = self.minprice

                    bidtotal = 0
                    offertotal = 0
                    slopetotal = 0

                    for agent in other_agents:
                        demand, slope = agent.helper_update_demand(trialprice, market.r)
                        slopetotal += slope
                        if demand > 0:
                            bidtotal += demand
                        elif demand < 0:
                            offertotal -= demand

                    rule_demand, rule_slope = rule.update_demand_rule(market.r, trialprice, agents[i].risk,
                                                                      agents[i]._mincash,
                                                                      agents[i].cash, agents[i]._minholding,
                                                                      agents[i]._maxdemand)
                    slopetotal += rule_slope
                    if rule_demand > 0:
                        bidtotal += rule_demand
                    elif rule_demand < 0:
                        offertotal -= rule_demand

                    if bidtotal > offertotal:
                        volume = offertotal
                    else:
                        volume = bidtotal

                    if bidtotal > 0.:
                        bidfrac = volume / bidtotal
                    else:
                        bidfrac = 0.

                    if offertotal > 0.:
                        offerfrac = volume / offertotal
                    else:
                        offerfrac = 0.

                    scount += 1

                rule.current_clearing_price = trialprice
                tp = self.taupnew * (rule.current_clearing_price - rule.previous_clearing_price + market.dividend)
                rule.rule_profit = self.taupdecay * rule.rule_profit + tp * rule.rule_holding

                # previous_holding = rule.rule_holding

                if rule_demand > 0:
                    rule.rule_holding += rule_demand * bidfrac
                    # Used in other profit setup
                    # rule.rule_cleared.put(rule_demand * bidfrac)

                else:
                    rule.rule_holding += rule_demand * offerfrac
                    # rule.rule_cleared.put(rule_demand * offerfrac)

                # rule.rule_profit = previous_holding * (rule.current_clearing_price -
                #                                        rule.previous_clearing_price + market.dividend) - \
                #                    (rule.previous_clearing_price * rule.rule_cleared.get())