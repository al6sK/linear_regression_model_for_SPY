import yfinance as yf
import time
# USA markets
spy = yf.Ticker("SPY")
df = spy.history(start="1993-01-29", interval="1d")
df.to_csv("data/USA_Markets/SPY_2025.csv")

sp500 = yf.Ticker("^GSPC")
df = sp500.history(start="1993-01-29", interval="1d")
df.to_csv("data/USA_Markets/S&P500_2025.csv")

nasdaq = yf.Ticker("^IXIC")
df = nasdaq.history(start="1993-01-29", interval="1d")
df.to_csv("data/USA_Markets/NASDAQ_2025.csv")

dji = yf.Ticker("^DJI")
df = dji.history(start="1993-01-29", interval="1d")
df.to_csv("data/USA_Markets/Dji_2025.csv")

# Europe markets
cac40 = yf.Ticker("^FCHI")
df = cac40.history(start="1993-01-29", interval="1d")
df.to_csv("data/EU_Markets/CAC40_2025.csv")

daxi = yf.Ticker("^GDAXI")
df = daxi.history(start="1993-01-29", interval="1d")
df.to_csv("data/EU_Markets/DAX_2025.csv")
 
# Asian markets
aord = yf.Ticker("^AORD")
df = aord.history(start="1993-01-29", interval="1d")
df.to_csv("data/Asian_Markets/Aord_2025.csv")

hsi = yf.Ticker("^HSI")
df = hsi.history(start="1993-01-29", interval="1d")
df.to_csv("data/Asian_Markets/HSI_2025.csv")
 
nikkei = yf.Ticker("^N225")
df = nikkei.history(start="1993-01-29", interval="1d")
df.to_csv("data/Asian_Markets/Nikkei_2025.csv")

