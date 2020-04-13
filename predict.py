from model import TradingModel
from termcolor import cprint

import dataManager
import settings

SYMBOL = 'NCLH'
DO_TRADING = True

if __name__ == '__main__':
    cprint('========= Training Model =========', 'blue')
    cprint(' / Loading Data Manager', 'green')
    env = dataManager.stockEnv(SYMBOL, DO_TRADING)
    cprint(' / Loading Model', 'green')
    model = TradingModel()
    cprint(' / Q Learn Time!', 'green')
    model.QLearn(env, settings.EPOCHS, SYMBOL)
    
