import tdameritrade

# Minimum amout of money in account allowed
MIN_CASH_VAL = 100

class TradeManager:
    tdclient: tdameritrade.TDClient
    def __init__(self, tdclient: tdameritrade.TDClient):
        self.tdclient = tdclient
    
    def make