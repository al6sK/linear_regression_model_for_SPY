import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
import numpy as np

warnings.filterwarnings("ignore")

aord= pd.read_csv('data/Asian_Markets/Aord_2025.csv')
hsi = pd.read_csv('data/Asian_Markets/HSI_2025.csv')
nikkei = pd.read_csv('data/Asian_Markets/Nikkei_2025.csv')

daxi = pd.read_csv('data/EU_Markets/DAX_2025.csv')
cac40 = pd.read_csv('data/EU_Markets/CAC40_2025.csv')

dji = pd.read_csv('data/USA_Markets/Dji_2025.csv')
nasdaq = pd.read_csv('data/USA_Markets/NASDAQ_2025.csv')
sp500 = pd.read_csv('data/USA_Markets/S&P500_2025.csv')
spy = pd.read_csv('data/USA_Markets/SPY_2025.csv')

#print(spy.tail(5))

indicepanel = pd.DataFrame(index=spy.index)
indicepanel['spy'] = spy['Open'].shift(-1) - spy['Open']
indicepanel['spy_lag1'] = indicepanel['spy'].shift(1)
indicepanel['sp500'] = sp500['Open'] - sp500['Open'].shift(1)
indicepanel['nasdaq'] = nasdaq['Open'] - nasdaq['Open'].shift(1)
indicepanel['dji'] = dji['Open'] - dji['Open'].shift(1)

indicepanel['cac40'] = cac40['Open'] - cac40['Open'].shift(1)
indicepanel['daxi'] = daxi['Open'] - daxi['Open'].shift(1)

indicepanel['aord'] = aord['Close'] - aord['Open'] 
indicepanel['hsi'] = hsi['Close'] - hsi['Open']
indicepanel['nikkei'] = nikkei['Close'] - nikkei['Open']

indicepanel['price'] = spy['Open']

#print(indicepanel.head(5))

indicepanel = indicepanel.ffill()
indicepanel = indicepanel.dropna()

#print(indicepanel.isnull().sum())

#print(indicepanel.shape)

#_________________________
# data splitting
#_________________________

Train = indicepanel.iloc[-2000:-1000,:].copy()
Test = indicepanel.iloc[-1000:,:].copy()
#print(Train.shape, Test.shape)

sm = scatter_matrix(Train, alpha=0.2, figsize=(10, 10))
plt.savefig('plots/scatter_matrix.png', dpi=300)
plt.close()

#_________________________
# Regression Model
#_________________________

formula = 'spy ~ spy_lag1 + sp500 + nasdaq + dji + cac40 + daxi + aord + hsi + nikkei'
lm = smf.ols(formula=formula, data=Train).fit()

#print(lm.summary())

Train['PredictedY'] = lm.predict(Train)
Test['PredictedY'] = lm.predict(Test)


# Scatter plot
plt.figure()
plt.scatter(Train['spy'], Train['PredictedY'], alpha=0.2)
plt.title('Train Data: Actual vs Predicted')
plt.xlabel('Actual SPY')
plt.ylabel('Predicted SPY')

# save plot
plt.savefig('plots/train_scatter.png', dpi=300)
plt.close()

# RMSE - Root Mean Squared Error, Adjusted R^2
def adjustedMetric(data, model, model_k, yname):
    data['yhat'] = model.predict(data)
    SST = ((data[yname] - data[yname].mean())**2).sum()
    SSR = ((data['yhat'] - data[yname].mean())**2).sum()
    SSE = ((data[yname] - data['yhat'])**2).sum()
    r2 = SSR/SST
    adjustR2 = 1 - (1-r2)*(data.shape[0] - 1)/(data.shape[0] -model_k -1)
    RMSE = (SSE/(data.shape[0] -model_k -1))**0.5
    return adjustR2, RMSE

def assessTable(test, train, model, model_k, yname):
    r2test, RMSEtest = adjustedMetric(test, model, model_k, yname)
    r2train, RMSEtrain = adjustedMetric(train, model, model_k, yname)
    assessment = pd.DataFrame(index=['R2', 'RMSE'], columns=['Train', 'Test'])
    assessment['Train'] = [r2train, RMSEtrain]
    assessment['Test'] = [r2test, RMSEtest]
    return assessment

print(assessTable(Test , Train , lm , 9 , 'spy').head(2))

#_________________________
# Strategy Implementation on Train Data
#_________________________
Train['Order'] =[1 if sig > 0 else -1 for sig in Train['PredictedY']]
Train['Profit'] = Train['Order'] * Train['spy']
# Calculate cumulative wealth
Train['Wealth'] = Train['Profit'].cumsum()
print("Total profit made in train: ", Train['Profit'].sum())
# Calculate Buy and Hold strategy
Train['BuyHold'] = Train['price'] - Train['price'].iloc[0]
print("Buy and Hold profit in train: ", Train['price'].iloc[-1] - Train['price'].iloc[0])


plt.figure(figsize=(10,10))

plt.plot(Train['Wealth'], label='Strategy Wealth', color='blue')
plt.plot(Train['BuyHold'], label='Buy & Hold', color='green')

plt.title('Cumulative Returns: Strategy vs Buy & Hold')
plt.xlabel('Time')
plt.ylabel('Profit')
plt.legend()
plt.grid(True)

plt.savefig('plots/strategy_vs_buyhold_on_Train_Data.png', dpi=300)
plt.close()

#_________________________
# Strategy Implementation on Test Data
#_________________________
Test['Order'] =[1 if sig > 0 else -1 for sig in Test['PredictedY']]
Test['Profit'] = Test['Order'] * Test['spy']
# Calculate cumulative wealth
Test['Wealth'] = Test['Profit'].cumsum()
print("Total profit made in Test: ", Test['Profit'].sum())
# Calculate Buy and Hold strategy
Test['BuyHold'] = Test['price'] - Test['price'].iloc[0]
print("Buy and Hold profit in Test: ", Test['price'].iloc[-1] - Test['price'].iloc[0])


plt.figure(figsize=(10,10))

plt.plot(Test['Wealth'], label='Strategy Wealth', color='blue')
plt.plot(Test['BuyHold'], label='Buy & Hold', color='green')

plt.title('Cumulative Returns: Strategy vs Buy & Hold')
plt.xlabel('Time')
plt.ylabel('Profit')
plt.legend()
plt.grid(True)

plt.savefig('plots/strategy_vs_buyhold_on_Test_Data.png', dpi=300)
plt.close()

#_________________________
#sharpe ratio for Train and Test Data
#_________________________
Train['Wealth'] = Train['Wealth'] + Train.loc[Train.index[0],'price']
Test['Wealth'] = Test['Wealth'] + Test.loc[Test.index[0],'price']

Train['Return'] = np.log(Train['Wealth']) - np.log(Train['Wealth'].shift(1))
dailyr = Train['Return'].dropna()

print("")
print("daily sharpe ratio on Train data is :" , dailyr.mean() / dailyr.std(ddof=1) ) 
print("yearly sharpe ratio on Train data is :" , (252**0.5) * dailyr.mean() / dailyr.std(ddof=1) )

Test['Return'] = np.log(Test['Wealth']) - np.log(Test['Wealth'].shift(1))
dailyr = Test['Return'].dropna()

print("")
print("daily sharpe ratio on Test data is :" , dailyr.mean() / dailyr.std(ddof=1) ) 
print("yearly sharpe ratio on Test data is :" , (252**0.5) * dailyr.mean() / dailyr.std(ddof=1) )

#_________________________
# Maximum drawdown for Train and Test Data
#_________________________

Train['Peak'] = Train['Wealth'].cummax()
Train['Drawdown'] = (Train['Peak'] - Train['Wealth']) / Train['Peak']

Test['Peak'] = Test['Wealth'].cummax()
Test['Drawdown'] = (Test['Peak'] - Test['Wealth'] ) / Test['Peak']

print("")
print("Maximum drawdonw in train is ", Train['Drawdown'].max())
print("Maximum drawdonw in test is ", Test['Drawdown'].max())


