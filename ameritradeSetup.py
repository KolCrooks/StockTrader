import tdameritrade
import tdameritrade.auth

import os

info = tdameritrade.auth.authentication(os.getenv('TDAMERITRADE_CLIENT_ID'), "http://localhost:8080")

print(info)