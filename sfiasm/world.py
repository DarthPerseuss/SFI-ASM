from tqdm.auto import tqdm
from sfiasm.datahandler import universal_trend
import numpy as np
import matplotlib.pyplot as plt
from sfiasm.agents import TechnicalGA
from operator import attrgetter


def plot_animated_var(t, variable):

    fig, ax = plt.subplots()
    ax.scatter(t, variable)
    plt.clf()
    plt.show()


class WallStreet:

    def __init__(self, agents, learning_interval, bitmutation_rate, crossover_rate,
                 rr_ratio=None, rr_ratio_tech=None, disable_dividend=False):

        self._p_pointmutation = bitmutation_rate
        self._p_crossover = crossover_rate
        self._learning_interval = learning_interval
        self.agents = agents
        if rr_ratio is not None:
            self._rr_ratio = rr_ratio
        if rr_ratio_tech is not None:
            self._rr_ratio_tech = rr_ratio_tech

        self.trend_followers = [agent for agent in self.agents if 'trend' in agent.tag]
        self.tech_ga = [agent for agent in self.agents if agent.tag == 'tech_ga']
        self.ga_agents = [agent for agent in self.agents if agent.tag == 'ga']
        self.under_ga = [agent for agent in self.agents if agent.tag == 'under']
        self.noise = [agent for agent in self.agents if agent.tag == 'noise']

        self.all_ga = self.tech_ga + self.ga_agents + self.under_ga
        self.disable_dividend = disable_dividend

        if self.ga_agents:
            self._fundamental_length_ga = len([k for k in range(len(self.ga_agents[0].rule_keys))
                                               if self.ga_agents[0].rule_keys[k] < 32])
            self._technical_length_ga = len([k for k in range(len(self.ga_agents[0].rule_keys))
                                             if self.ga_agents[0].rule_keys[k] >= 32])
            if self._technical_length_ga == 0:
                self._is_fundamental = True
            else:
                self._is_fundamental = False

        else:
            self._fundamental_length_ga = len([k for k in range(len(self.under_ga[0].rule_keys))
                                               if self.under_ga[0].rule_keys[k] < 32])
            self._technical_length_ga = len([k for k in range(len(self.under_ga[0].rule_keys))
                                             if self.under_ga[0].rule_keys[k] >= 32])
            if self._technical_length_ga == 0:
                self._is_fundamental = True
            else:
                self._is_fundamental = False

        if self.tech_ga:
            self._technical_length_ga = len(self.tech_ga[0].rule_keys)
            self._fundamental_length_ga = 0

    def get_bitfrac(self, count_bits=False):

        fracs = {}
        spec_counts = {}
        one_count_fund = zero_count_fund = one_count_tech = zero_count_tech = 0
        if self.ga_agents:
            if self._is_fundamental:
                for agent in self.ga_agents:
                    for rule in agent.strategies:
                        if count_bits:
                            each_1_fund = rule.condition[:self._fundamental_length_ga].count(1)
                            each_0_fund = rule.condition[:self._fundamental_length_ga].count(0)
                            one_count_fund += each_1_fund
                            zero_count_fund += each_0_fund
                            agent.fundamental_frac += each_1_fund + each_0_fund

                        else:
                            agent.fundamental_frac += self._fundamental_length_ga - \
                                                  rule.condition[:self._fundamental_length_ga].count(2)

                ga_fundfrac = sum((agent.fundamental_frac for agent in self.ga_agents)) / \
                              (len(self.ga_agents) * self.ga_agents[0].numStrategies * self._fundamental_length_ga)

                for agent in self.ga_agents:
                    agent.fundamental_frac = 0
                fracs['fund'] = ga_fundfrac
                spec_counts['ga fund'] = (zero_count_fund, one_count_fund)

            else:
                for agent in self.ga_agents:
                    for rule in agent.strategies:
                        if count_bits:
                            each_1_tech = rule.condition[self._fundamental_length_ga:].count(1)
                            each_0_tech = rule.condition[self._fundamental_length_ga:].count(0)
                            one_count_tech += each_1_tech
                            zero_count_tech += each_0_tech
                            agent.technical_frac += each_1_tech + each_0_tech

                            each_1_fund = rule.condition[:self._fundamental_length_ga].count(1)
                            each_0_fund = rule.condition[:self._fundamental_length_ga].count(0)
                            one_count_fund += each_1_fund
                            zero_count_fund += each_0_fund
                            agent.fundamental_frac += each_1_fund + each_0_fund

                        else:
                            agent.fundamental_frac += self._fundamental_length_ga - \
                                                      rule.condition[:self._fundamental_length_ga].count(2)
                            agent.technical_frac += self._technical_length_ga - \
                                                    rule.condition[self._fundamental_length_ga:].count(2)

                    agent.total_frac = agent.fundamental_frac + agent.technical_frac

                ga_fundfrac = sum((agent.fundamental_frac for agent in self.ga_agents)) / \
                              (len(self.ga_agents) * self.ga_agents[0].numStrategies * self._fundamental_length_ga)
                ga_techfrac = sum((agent.technical_frac for agent in self.ga_agents)) / \
                              (len(self.ga_agents) * self.ga_agents[0].numStrategies * self._technical_length_ga)
                ga_totfrac = sum((agent.total_frac for agent in self.ga_agents)) / \
                             (len(self.ga_agents) * self.ga_agents[0].numStrategies *
                              (self._fundamental_length_ga + self._technical_length_ga))

                for agent in self.ga_agents:
                    agent.fundamental_frac = 0
                    agent.technical_frac = 0
                    agent.total_frac = 0

                fracs['ga'] = (ga_fundfrac, ga_techfrac, ga_totfrac)
                if count_bits:
                    spec_counts['ga fund'] = (zero_count_fund, one_count_fund)
                    spec_counts['ga tech'] = (zero_count_tech, one_count_tech)

        if self.tech_ga:
            for agent in self.tech_ga:
                for rule in agent.strategies:
                    agent.technical_frac += self._technical_length_ga - rule.condition.count(2)
                if agent.force_mutationdiv is True:
                    # Stop forcing when specificity reaches a threshold
                    if agent.technical_frac / (self._technical_length_ga * agent.numStrategies) >= 0.2:
                        agent.force_mutationdiv = False
                    else:
                        agent.force_mutationdiv = True
            techfrac = sum((agent.technical_frac for agent in self.tech_ga)) / \
                       (len(self.tech_ga) * self.tech_ga[0].numStrategies * 32)

            for agent in self.tech_ga:
                agent.technical_frac = 0
            fracs['tech'] = techfrac

        if self.under_ga:
            one_count_fund = zero_count_fund = one_count_tech = zero_count_tech = 0
            for agent in self.under_ga:
                for rule in agent.strategies:
                    if count_bits:
                        each_1_tech = rule.condition[self._fundamental_length_ga:].count(1)
                        each_0_tech = rule.condition[self._fundamental_length_ga:].count(0)
                        one_count_tech += each_1_tech
                        zero_count_tech += each_0_tech
                        agent.technical_frac += each_1_tech + each_0_tech

                        each_1_fund = rule.condition[:self._fundamental_length_ga].count(1)
                        each_0_fund = rule.condition[:self._fundamental_length_ga].count(0)
                        one_count_fund += each_1_fund
                        zero_count_fund += each_0_fund
                        agent.fundamental_frac += each_1_fund + each_0_fund

                    else:
                        agent.fundamental_frac += self._fundamental_length_ga - \
                                                  rule.condition[:self._fundamental_length_ga].count(2)
                        agent.technical_frac += self._technical_length_ga - \
                                                rule.condition[self._fundamental_length_ga:].count(2)

                agent.total_frac = agent.fundamental_frac + agent.technical_frac

            under_fundfrac = sum((agent.fundamental_frac for agent in self.under_ga)) / \
                          (len(self.under_ga) * self.under_ga[0].numStrategies * self._fundamental_length_ga)
            under_techfrac = sum((agent.technical_frac for agent in self.under_ga)) / \
                          (len(self.under_ga) * self.under_ga[0].numStrategies * self._technical_length_ga)
            under_totfrac = sum((agent.total_frac for agent in self.under_ga)) / \
                         (len(self.under_ga) * self.under_ga[0].numStrategies *
                          (self._fundamental_length_ga + self._technical_length_ga))

            for agent in self.under_ga:
                agent.fundamental_frac = 0
                agent.technical_frac = 0
                agent.total_frac = 0
            fracs['under'] = (under_fundfrac, under_techfrac, under_totfrac)
            if count_bits:
                spec_counts['under fund'] = (zero_count_fund, one_count_fund)
                spec_counts['under tech'] = (zero_count_tech, one_count_tech)

        if count_bits:
            return fracs, spec_counts
        else:
            return fracs

    def get_variance(self):
        variances = []
        for agent in self.trend_followers:
            variances.append(agent.variance)
        return variances

    def get_profits(self):
        profit_dict = {}
        if self.ga_agents:
            profit_dict['ga'] = [agent.profit for agent in self.ga_agents]
        if self.trend_followers:
            profit_dict['trend'] = [agent.profit for agent in self.trend_followers]
        if self.tech_ga:
            profit_dict['tech'] = [agent.profit for agent in self.tech_ga]
        if self.under_ga:
            profit_dict['under'] = [agent.profit for agent in self.under_ga]
        if self.noise:
            profit_dict['noise'] = [agent.profit for agent in self.noise]

        return profit_dict

    def get_holdings(self):
        holding_dict = {}
        if self.ga_agents:
            holding_dict['ga'] = [agent.holding for agent in self.ga_agents]
        if self.trend_followers:
            holding_dict['trend'] = [agent.holding for agent in self.trend_followers]
        if self.tech_ga:
            holding_dict['tech'] = [agent.holding for agent in self.tech_ga]
        if self.under_ga:
            holding_dict['under'] = [agent.holding for agent in self.under_ga]
        if self.noise:
            holding_dict['noise'] = [agent.holding for agent in self.noise]

        return holding_dict

    def get_strengths(self):
        strength_dict = {}
        if self.ga_agents:
            strength_dict['ga'] = [agent.average_strength() for agent in self.ga_agents]
        if self.tech_ga:
            strength_dict['tech'] = [agent.average_strength() for agent in self.tech_ga]
        if self.under_ga:
            strength_dict['under'] = [agent.average_strength() for agent in self.under_ga]
        return strength_dict

    def get_variable(self, variable: str):
        return [getattr(agent, variable) for agent in self.agents]

    def get_demands(self):
        all_demands = dict()
        if self.ga_agents:
            ga_demands = {'ga': [agent.demand for agent in self.ga_agents]}
            all_demands.update(ga_demands)
        if self.under_ga:
            under_demands = {'under': [agent.demand for agent in self.under_ga]}
            all_demands.update(under_demands)
        if self.tech_ga:
            tech_demands = {'tech': [agent.demand for agent in self.tech_ga]}
            all_demands.update(tech_demands)
        if self.trend_followers:
            trend_demands = {'trend': [agent.demand for agent in self.trend_followers]}
            all_demands.update(trend_demands)
        if self.noise:
            all_demands['noise'] = [agent.profit for agent in self.noise]

        return all_demands

    def get_cash(self):
        cash_dict = {}
        if self.ga_agents:
            cash_dict['ga'] = [agent.cash for agent in self.ga_agents]
        if self.trend_followers:
            cash_dict['trend'] = [agent.cash for agent in self.trend_followers]
        if self.tech_ga:
            cash_dict['tech'] = [agent.cash for agent in self.tech_ga]
        if self.under_ga:
            cash_dict['under'] = [agent.cash for agent in self.under_ga]
        if self.noise:
            cash_dict['noise'] = [agent.cash for agent in self.noise]

        return cash_dict

    def get_bestforecast_params(self):
        best_forecast_params = dict()
        if self.ga_agents:
            best_fcast = [min(agent.activeList, key=attrgetter('actVar')) for agent in self.ga_agents]
            ga_bestfcast = {'ga': [(best_f.pdcoeff, best_f.offset) for best_f in best_fcast]}

            best_forecast_params.update(ga_bestfcast)
        if self.under_ga:
            best_fcast = [min(agent.activeList, key=attrgetter('actVar')) for agent in self.under_ga]
            under_bestfcast = {'under': [(best_f.pdcoeff, best_f.offset) for best_f in best_fcast]}
            best_forecast_params.update(under_bestfcast)

        if self.tech_ga:
            best_fcast = [min(agent.activeList, key=attrgetter('actVar')) for agent in self.tech_ga]
            tech_bestfcast = {'tech': [(best_f.pdcoeff, best_f.offset) for best_f in best_fcast]}
            best_forecast_params.update(tech_bestfcast)
        return best_forecast_params

    def get_activerule_params(self):
        active_forecast_params = dict()
        if self.ga_agents:
            ga_activefcast = {'ga': [([active.pdcoeff for active in agent.activeList],
                                      [active.offset for active in agent.activeList])
                                     for agent in self.ga_agents]}
            active_forecast_params.update(ga_activefcast)

        if self.under_ga:
            under_activefcast = {'under': [([active.pdcoeff for active in agent.activeList],
                                            [active.offset for active in agent.activeList])
                                           for agent in self.under_ga]}
            active_forecast_params.update(under_activefcast)

        if self.tech_ga:
            tech_activefcast = {'tech': [([active.pdcoeff for active in agent.activeList],
                                          [active.offset for active in agent.activeList])
                                         for agent in self.tech_ga]}
            active_forecast_params.update(tech_activefcast)
        return active_forecast_params

    def trade(self, steps, market, specialist, divplusprice_trend=False, disable_tqdm=False,
              universaltrend=None, use_bitconditions=True, tau=None, statprint_interval=1000, print_fitness=True,
              save_volume=True, save_specificfracs=False, frequency_specific_frac=100, frequency_strength=100,
              save_specbit_count=False, save_trendvars=False, save_demands=False, save_nactive=False, save_profits=False,
              save_strength=False, save_holding=False, save_cash=False, save_price=False, save_div=False, save_bestfcast=False,
              save_activefcast=False, save_agents=False, set_dividends=None, ):

        """
        This function is a master function to handle a lot of tasks.

        :param steps:
        :param market:
        :param specialist:
        :param disable_tqdm:
        :param universaltrend:
        :param tau:
        :param statprint_interval:
        :param print_fitness:
        :param save_volume:
        :param save_specificfracs:
        :param save_trendvars:
        :param save_demands:
        :param save_nactive:
        :param save_profits:
        :param save_strength:
        :param save_holding:
        :return:
        """

        specific_fracs = []
        trend_vars = []
        num_active = []
        profits = []
        avg_strength = []
        demands = []
        volume_series = []
        agent_holdings = []
        agent_cash = []
        specbit_count = []
        best_fcastparams = []
        active_fcastparams = []
        all_agents = None
        prices = None
        dividends = None

        is_parent_profit = False
        if self.tech_ga:
            for base in TechnicalGA.__bases__:
                if str(base.__name__) == 'ProfitGA':
                    is_parent_profit = True

        episode = 0

        for step in tqdm(range(steps), disable=disable_tqdm):

            if set_dividends is not None:
                market.copy_arprocess(set_dividends[step+501])
            else:
                market.advance_arprocess()

            market.currentTime += 1

            for agent in self.agents:
                agent.creditearnings_paytaxes(market.dividend, market.price, market.r)

            if use_bitconditions:
                market.determine_marketcondition_bitconditions()

            else:
                if len(market.marketCondition) == 32:
                    if divplusprice_trend is True:
                        market.determine_divandpricetrends()
                    else:
                        market.determine_technical()

            if universaltrend is not None:
                if tau is not None:
                    if len(tau) == 1:
                        u_variance, u_pdcoeff, u_offset = universal_trend(market, tau)
                        for trend in self.trend_followers:
                            trend.variance = u_variance
                            trend.pdcoeff = u_pdcoeff
                            trend.offset = u_offset
                    else:
                        # the trend_followers should be a nested list
                        for i, horizon in enumerate(tau):
                            u_variance, u_pdcoeff, u_offset = universal_trend(market, horizon)
                            for trend in self.trend_followers[i]:
                                trend.variance = u_variance
                                trend.pdcoeff = u_pdcoeff
                                trend.offset = u_offset

            for agent in self.agents:
                if agent.tag == 'ga' or agent.tag == 'under':
                    agent.prepare_for_trading(1 / self._learning_interval, self._p_pointmutation,
                                              self._p_crossover, market, removal_replacement_ratio=self._rr_ratio)
                elif 'tech' in agent.tag:
                    agent.prepare_for_trading(1 / self._learning_interval, self._p_pointmutation,
                                              self._p_crossover, market, removal_replacement_ratio=self._rr_ratio_tech)

                elif agent.tag == 'noise' or agent.tag == 'herder':
                    agent.randomize_parameters()

            if save_specificfracs and not save_specbit_count:
                if step % frequency_specific_frac == 0:
                    specific_fracs.append(self.get_bitfrac())

            elif save_specificfracs and save_specbit_count:
                if step % frequency_specific_frac == 0:
                    specific_fracs.append(self.get_bitfrac(count_bits=True))

            if step % statprint_interval == 0 and step != 0:
                if print_fitness:
                    episode += 1
                    print("mean fitness: {:.4f} (episode {})".format(np.mean([agent.average_strength()
                                                                              for agent in self.all_ga]), episode))

            if save_strength:
                if step % frequency_strength:
                    avg_strength.append(self.get_strengths())

            # if self.under_ga:
            #     for agent in self.under_ga:
            #         agent.update_errorterm()

            specialist.clear_market(self.agents, market)

            if save_volume:
                volume_series.append(specialist.volume)

            if save_holding:
                agent_holdings.append(self.get_holdings())

            if save_cash:
                agent_cash.append(self.get_cash())

            if save_demands:
                demands.append(self.get_demands())

            if save_trendvars:
                trend_vars.append(self.get_variance())

            if save_bestfcast:
                best_fcastparams.append(self.get_bestforecast_params())

            if save_activefcast:
                if step % frequency_specific_frac == 0:
                    active_fcastparams.append(self.get_activerule_params())

            if save_nactive:
                if self.ga_agents:
                    num_active.append([len(agent.activeList) for agent in self.ga_agents])
                if self.tech_ga:
                    num_active.append([len(agent.activeList) for agent in self.tech_ga])
                if self.under_ga:
                    num_active.append([len(agent.activeList) for agent in self.under_ga])

            if save_profits:
                profits.append(self.get_profits())

            for agent in self.ga_agents:
                agent.update_performance(market)

            if self.under_ga:
                for agent in self.under_ga:
                    agent.update_performance(market)

            if self.tech_ga:
                for agent in self.tech_ga:
                    # Updates active variances of the agents
                    agent.update_performance(market)

                    # Rule profit is EMA profit
                    if is_parent_profit:
                        agent.bestForecast.rule_profit = agent.profit

            # Updates the profit and hence performance of technical GA agents.
            if is_parent_profit:
                specialist.determine_profit_and_update_performances(self.tech_ga, market)

        # The below are used only in multiprocessing
        if save_price:
            prices = market.priceTimeSeries

        if save_div:
            dividends = market.divTimeSeries

        if save_agents:
            if self.ga_agents:
                all_agents['ga'] = self.ga_agents
            if self.under_ga:
                all_agents['under'] = self.under_ga
            if self.tech_ga:
                all_agents['tech'] = self.tech_ga

        return_dict = {'specific fracs': specific_fracs, 'specific bit count': list(filter(None, specbit_count)),
                       'trend vars': trend_vars, 'demands': demands, 'holdings': agent_holdings, 'cash': agent_cash,
                       'num active': num_active, 'agent profits': profits, 'agent strengths': avg_strength,
                       'volume': volume_series,'prices': prices, 'dividends': dividends,
                       'best forecast': best_fcastparams, 'active forecasts': active_fcastparams, 'agents': all_agents}

        return return_dict


def trade_ree(steps, market, specialist):
    for step in range(steps):
        market.advance_arprocess()
        specialist.determine_price(agents=None, market=market, mode='ree')


def determine_ree_price(divseries):
    prices = [6.333 * dividend + 16.6882 for dividend in divseries]
    return prices
