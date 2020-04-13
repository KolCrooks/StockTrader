import os
# import typefile

from typing import List, Any
import time

import gym
import gym.spaces as spaces

import tdameritrade
import tdameritrade.auth

from termcolor import cprint, colored

quote_cols = ['52WkHigh', '52WkLow', 'lastPrice', 'volatility']

# Just for reference
ref_state_cols = ['52WkHigh', '52WkLow', 'lastPrice', 'volatility', 'P/L', 'gainMult']


N_DISCRETE_ACTIONS = 3 # BUY, HOLD, SELL
N_DISCRETE_STATES = len(ref_state_cols)


class stockEnv(gym.Env):
    __symbol = ""

    __last_state = []
    __last_quote = {}
    __buy_price = -1
    tdclient: tdameritrade.TDClient

    def __init__(self, stock_symbol: str):
        super(stockEnv, self).__init__()

        client_id = os.getenv('TDAMERITRADE_CLIENT_ID')
        account_id = os.getenv('TDAMERITRADE_ACCOUNT_ID')
        refresh_token = os.getenv('TDAMERITRADE_REFRESH_TOKEN')

        refresh_data = tdameritrade.auth.refresh_token(refresh_token, client_id)

        self.tdclient = tdameritrade.TDClient(refresh_data['access_token'], [account_id])

        # Stock settings
        self.__symbol = stock_symbol

        # Define action and observation space
        self.__action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.__observation_space = spaces.Discrete(N_DISCRETE_STATES)
        
        self.reset()



    def cleanQuoteInstance(self, quote: Any, last_quote: Any):
        """
        Turns quote data from tdclient into usable array based data
    
        This function is made to convert quote data into data that can be used in tensorflow

        To get stock objects, use code:
        tdclient.quote(symbol=stock_symbol)

        Parameters: 
        quote (int): Current quote object for a given stock
        last_quote (int): last quote object for a given stock
    
        Returns: 
        List[float]: List of data 
        """
        cleaned = []
        # Get Changes
        last_val = last_quote['lastPrice']
        curr_val = quote['lastPrice']

        # Add the quote info
        for col in quote_cols:
            cleaned.append(quote[col])

        # Change the cleaned values to % based
        cleaned[0] = (cleaned[0]-curr_val) / curr_val # 52 Week High
        cleaned[1] = (cleaned[1]-curr_val) / curr_val # 52 Week Low
        cleaned[2] = (cleaned[2]-curr_val) / last_val # Price Change

        cleaned.extend([0, 0])

        if self.__buy_price == -1:
            cleaned[4] = 0 # Update the P/L value
        else:
            cleaned[4] = (self.__buy_price-curr_val) / curr_val

        return cleaned

    def calcReward(self, state: List[float], action: int):
        """
        Calculates the reward of a given action based on a given state
    
        Parameters: 
        state (List[float]): Description of arg1 
        action (int): Action id (0-2)

        Returns: 
        (lambda x, int, int): Returns P/L changer, Gains Multiplier, Reward
        """
        if action == 0: # HOLD
            return (lambda x: x, 1, 0)
        elif action == 1: # SELL
            self.__buy_price = -1 # Reset Buy Price
            return (lambda x: 0, 0, state[4] * state[5]) # P/L Percent * Gain Multiplier
        else: # BUY
            return (lambda x: (x/2), 2, -1); # TODO Change this to something smarter
 
    def step(self, action: int):
        
        pl_func, gain_mul, reward = self.calcReward(self.__last_state, action)

        self.__last_state[5] *= gain_mul # update gain multiplier

        n_quote = self.tdclient.quote(self.__symbol)[self.__symbol]

        if action == 2: # Update Buy price if bought
            if self.__buy_price == -1:
                self.__buy_price = n_quote['lastPrice']
            self.__buy_price = (self.__buy_price + n_quote['lastPrice']) / 2

        new_state = self.cleanQuoteInstance(self.__last_quote, n_quote)
        new_state[4] = pl_func(new_state[4])

        return new_state, reward, False, (self.__buy_price)

    def reset(self):
        # Seed the initial Data
        lastQuote = self.tdclient.quote(self.__symbol)[self.__symbol]
        print('Waiting 2s to seed the price change...')
        time.sleep(2)
        newQuote = self.tdclient.quote(self.__symbol)[self.__symbol]

        self.__last_state = self.cleanQuoteInstance(lastQuote, newQuote)
        self.__last_quote = newQuote

        return self.__last_state
    
    def render(self, mode='human'):
        label = lambda x: cprint(x, 'red', 'on_cyan')
        cprint('================================', 'grey')
        headers = ["Symbol", "Current P/L", "Reward Multiplier"]
        data    = [self.__symbol, self.__last_state[4], self.__last_state[4]]

        header_print = ""
        data_print   = ""

        for i in range(len(headers)):
            header_print += headers[i] + ("  ")
            data_print += str(data[i]) + (' ' * (len(headers[i]) - len(str(data[i])))) + ("   ")

        cprint(header_print, 'cyan')
        cprint(data_print, 'green')

        cprint('================================', 'grey')
        print()

    def state(self):
        return self.__last_state
