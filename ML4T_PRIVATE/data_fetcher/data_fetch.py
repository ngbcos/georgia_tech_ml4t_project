

#https://query1.finance.yahoo.com/v7/finance/download/NVDA?period1=1368342000&period2=1526108400&interval=1d&events=history&crumb=JgXJUmQoi/J


import requests
import time
import datetime

start_date = "05/12/2013"
end_date = str(datetime.date.today())

start_date_unix = int(datetime.datetime.strptime(start_date, "%m/%d/%Y").strftime("%s"))
end_date_unix = int(datetime.datetime.strptime(end_date, "%Y-%m-%d").strftime("%s"))

stocks = ["NVDA", "MSFT"]

for stock in stocks:
    url_string_prefix = "https://query1.finance.yahoo.com/v7/finance/download/"
    url_string_suffix = "?period1=" + str(start_date_unix) + "&period2=" + str(end_date_unix) + "&interval=1d&events=history&crumb=JgXJUmQoi/J"
    url_string = url_string_prefix + url_string_suffix

    print(url_string)
