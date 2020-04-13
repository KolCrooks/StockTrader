import os
# import typefile

from typing import List, Any

import time
from datetime import datetime

import gym
import gym.spaces as spaces

import tdameritrade
import tdameritrade.auth

from termcolor import cprint, colored
from tabulate import tabulate

import settings

quote_cols = ['52WkHigh', '52WkLow', 'lastPrice', 'volatility']

# Just for reference
ref_state_cols = ['52WkHigh', '52WkLow', 'lastPrice', 'volatility', 'P/L', 'gainMult']

actions = ['HOLD', 'SELL', 'BUY']


N_DISCRETE_ACTIONS = len(actions)
N_DISCRETE_STATES = len(ref_state_cols)


class stockEnv(gym.Env):
    __symbol = ""

    __last_state = []
    __last_quote = {}
    __buy_price = -1
    __net_profit = 0
    __start_price = 0

    tdclient: tdameritrade.TDClient

    def __init__(self, stock_symbol: str, DO_TRADING=False):
        super(stockEnv, self).__init__()
        
        # Login to TD Ameritrade
        self.refresh_tdClient()

        # Stock settings
        self.__symbol = stock_symbol

        # Define action and observation space
        self.__action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.__observation_space = spaces.Discrete(N_DISCRETE_STATES)
        
        self.reset()

    def refresh_tdClient(self):
        # Get account details from .env file
        client_id = os.getenv('TDAMERITRADE_CLIENT_ID')
        account_id = os.getenv('TDAMERITRADE_ACCOUNT_ID')
        refresh_token = os.getenv('TDAMERITRADE_REFRESH_TOKEN')

        # Refresh Access token
        refresh_data = tdameritrade.auth.refresh_token(refresh_token, client_id)

        # Set up client
        self.tdclient = tdameritrade.TDClient(refresh_data['access_token'], [account_id])

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

        # Add entries for P/L and Reward Multiplier
        cleaned.extend([0, 0])

        # Calculate the Profit/Loss value
        if self.__buy_price == -1:
            cleaned[4] = 0
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

        # TODO Change this to something smarter

        if action == 0: # HOLD
            return (1, -0.5)
        elif action == 1: # SELL
            # Cacluate Base R
            r = state[4] * state[5] * 2
            
            # Discout selling at 0 gain multiplier
            if state[5] == 0:
                # Equal to HOLD reward
                r = -0.5
            elif r > 0:
                # We want the AI to really like profits
                r *= 5

            return (0, r) # P/L Percent * Gain Multiplier
        else: # BUY
            # Buying again and again is bad because it increases risk
            # TODO we can probably change this
            return (2, -0.25); 

    def step(self, action: int):
        # Calculate the reward and gain multiplier
        gain_mul, reward = self.calcReward(self.__last_state, action)

        # Adjust new gains with the gain multiplier
        self.__last_state[5] *= gain_mul # update gain multiplier
        
        # Quote is sometimes broken so keep trying it until it works
        while True:
            try:
                q = self.tdclient.quote(self.__symbol)
                if self.__symbol in q:
                    n_quote = q[self.__symbol]
                    break
            except:
                # Sometimes the reason is that you get logged out of your account so refresh the client to fix the problem
                self.refresh_tdClient()


        if action == 1 and self.__buy_price != -1: 
            # update net profit
            self.__net_profit += (n_quote['lastPrice'] - self.__buy_price)
            self.__buy_price = -1 # Reset Buy Price

        elif action == 2: # Update Buy price if bought
            if self.__buy_price == -1: # Doesn't have any shares if == -1
                self.__buy_price = n_quote['lastPrice']
                self.__last_state[5] = 1 # Reset Multiplier
            # Update the buy price to be the mean of the values
            self.__buy_price = (self.__buy_price + n_quote['lastPrice']) / 2

        # Generate new state from quote difference
        new_state = self.cleanQuoteInstance(self.__last_quote, n_quote)

        #transfer multiplier to new state
        new_state[5] = self.__last_state[5]

        self.__last_state = new_state
        self.__last_quote = n_quote

        # Shut AI down at 4:00pm because the market closes at that time
        now = datetime.now().time()
        done = (now.hour == 12 + 4)

        # Return the new state, the reward, 
        return new_state, reward, done, (self.__buy_price)

    def reset(self):
        # Seed the initial Data
        lastQuote = self.tdclient.quote(self.__symbol)[self.__symbol]

        # Set the start price to be the price when the AI is started
        self.__start_price = lastQuote['lastPrice']

        # Wait n sec so that the market value can change
        print(f'Waiting {settings.TRADE_INTERVAL}s to seed the price change...')
        time.sleep(settings.TRADE_INTERVAL)
        newQuote = self.tdclient.quote(self.__symbol)[self.__symbol]

        # Initialize some variables quote state
        self.__last_state = self.cleanQuoteInstance(lastQuote, newQuote)
        self.__last_quote = newQuote

        return self.__last_state
    
    def render(self, mode='human'):
        
        # Find the difference in start price and current price
        price_diff = self.__start_price - self.__last_quote['lastPrice']

        # Create data Table
        headers = ["Symbol"     , "Current P/L"       , "Reward Multiplier" , "Buy Price"     , "Current Price", "Net Profit", "Net Change"]
        data    = [self.__symbol, f'{float(self.__last_state[4]):.3}', self.__last_state[5], self.__buy_price, self.__last_quote['lastPrice'], f'{float(self.__net_profit*settings.SHARES):.6}', f'{float(price_diff*settings.SHARES):.6}']
        table = tabulate([headers, data])
        cprint(table, 'cyan')

        # Add seperator
        cprint('================================', 'grey')
        print()

    def state(self):
        return self.__last_state
