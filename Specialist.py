from Stocks import Stock
import math


class Specialist(Stock):

    def __init__(self, max_stockprice, min_stockprice, min_excess, eta, max_iterations, num_shares,
                 init_price, dividend_startvalue, rho, noise_sd, d_bar, interest_rate):
        super().__init__(num_shares, init_price, dividend_startvalue, rho, noise_sd, d_bar, interest_rate)
        self.maxprice = max_stockprice
        self.minprice = min_stockprice
        self.initialPrice = init_price
        self.eta = eta
        self.minExcess = min_excess
        self.maxIterations = max_iterations
        self.eta = eta
        self.price = init_price
        self.priceTimeSeries = [init_price]

    def set_price(self, value):
        self.priceTimeSeries.append(value)
        self.price = value

    def clear_market(self, agents, pricesetting='auction'):
        done = False
        scount = 0
        trialprice = self.price
        agent_demands = []
        num_agents = len(agents)
        slope_total = 0

        # Normalize bids to match the number of holdings by each ageant
        def normalize_bids(demands, p_trial):
            for idx in range(len(demands)):
                # restrict offers to the amount of shares agents have
                if demands[idx] < -agents[idx].holding:
                    demands[idx] = -agents[idx].holding
                # restrict bids to the amount of cash agents have
                elif demands[idx] * p_trial > agents[idx].cash:
                    demands[idx] = math.floor(agents[idx].cash / p_trial)

            normalization_numer = abs(sum([x for x in demands if x < 0]))
            normalization_denom = sum([x for x in demands if x > 0])
            for idx in range(len(demands)):
                if demands[idx] > 0:
                    demands[idx] = demands[idx] * normalization_numer / normalization_denom
            return demands

        if pricesetting == 'auction':
            is_rationed = False
            selected_rules = []
            for agent in agents:
                selected_rules.append(agent.select_demandrule())
            while scount < self.maxIterations and not done:
                for i in range(len(selected_rules)):
                    agent_demands.append(agents[i].demand(p_trial=trialprice, selected_rule=selected_rules[i])[0])

                if abs(sum(agent_demands)) <= self.minExcess:
                    done = True
                if slope_total != 0:
                    # The increase or decrease parameter needs to be adaptive and for that
                    # determining the partial derivatives helps in determining this adaptive
                    # parameter as they show the next periods direction
                    trialprice -= sum(agent_demands) / slope_total
                else:
                    trialprice *= 1 + self.eta * sum(agent_demands)

                slope_total = sum([agents[i].demand(p_trial=trialprice, selected_rule=selected_rules[i])[1] for i in
                                   range(num_agents)])
                if trialprice > self.maxprice:
                    trialprice = self.maxprice
                if trialprice < self.minprice:
                    trialprice = self.minprice

                scount += 1
                if scount < self.maxIterations - 1 and done is False:
                    agent_demands = []

            # This part implements tha rationing and sets holding and agents cash
            # TODO: is it correct or even necessary?
            if scount == self.maxIterations and done is False:
                is_rationed = True
                offers = [elem for elem in agent_demands if elem < 0]
                if not offers is True:
                    num_bidders = num_agents - len(offers)
                    rationed_bids = round(abs(sum(offers)) / num_bidders)
                    for i in range(num_agents):
                        if agent_demands[i] > 0:
                            agents[i].set_holding(rationed_bids)
                            agents[i].set_cash(-rationed_bids * trialprice)
                            agents[i].set_holding(agent_demands[i])
                            agents[i].set_cash(rationed_bids * trialprice)
                self.set_price(trialprice)

            if is_rationed is False:
                agent_demands = normalize_bids(agent_demands, trialprice)
                for i, agent in enumerate(agents):
                    agent.set_holding(agent_demands[i])
                    agent.set_cash(agent_demands[i] * trialprice)
                self.set_price(trialprice)
