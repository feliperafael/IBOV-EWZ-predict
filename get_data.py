import pandas as pd
import requests
import os
import io

#Getting IBOV(EWZ) data

#Unstable method - server from alphavantage.co is out very often
stock_url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=EWZ&outputsize=full&apikey=T6RRM9LEWYQ1Z8NC&datatype=csv"
raw_response = requests.get(stock_url).content
df = pd.read_csv(io.StringIO(raw_response.decode('utf-8')))

#Saving file
df.to_pickle('./df')
#Reading file
df = pd.read_pickle('./df')

df.to_csv('input/daily_adjusted_EWZ_now.csv')
