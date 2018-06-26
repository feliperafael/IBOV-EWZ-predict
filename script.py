import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn import neural_network
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime

def MLPRegressor(X_train, X_test, y_train, y_test):
    print("MLPRegressor")
    error = []
    model = []
    for i in range(10):
        model.append(neural_network.MLPRegressor(hidden_layer_sizes=(100, ),max_iter=1000))
        model[i].fit(X_train,y_train)
        y_pred = model[i].predict(X_test)
        error.append(np.sqrt(mean_squared_error(y_pred, y_test)))
        
    print("RMSE: %.5f" % np.mean(error))
    return model[np.argmin(error)]

def LinearRegression(X_train, X_test, y_train, y_test):
    print("LinearRegression")
    reg = linear_model.LinearRegression()
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    print("RMSE: %.5f" % np.sqrt(mean_squared_error(y_pred, y_test)))

    return reg

def show_plot(data):
    fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})
    ax1.set_ylabel('Closing Price ($)',fontsize=12)
    ax2.set_ylabel('Volume ($ bn)',fontsize=12)
    ax2.set_yticks([int('%d000000000'%i) for i in range(10)])
    ax2.set_yticklabels(range(10))
    ax1.set_xticks([datetime.date(i,j,1) for i in range(2000,2019) for j in [1,12]])
    ax1.set_xticklabels('')
    ax2.set_xticks([datetime.date(i,j,1) for i in range(2000,2019) for j in [1,12]])
    ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')    for i in range(2000,2019) for j in [1,12]], fontsize=6)
    ax1.plot(data['timestamp'].astype(datetime.datetime),data['open'])
    ax2.bar(data['timestamp'].astype(datetime.datetime).values, data['volume'].values)
    fig.tight_layout()
    plt.show()

def main():
    file = pd.read_csv('input/daily_adjusted_EWZ.csv')
    file = file.sort_values(by='timestamp', ascending=True)
    #show_plot(file)
    close_tomorrow = file['close'][1:]    
    close_tomorrow = pd.Series(close_tomorrow)
    #print    file['close']
    file = file[:-1] # remove last line 
    #print    file['close']
    print(close_tomorrow[:5])
    file['close_tomorrow'] = 0
    print("TAM CLOSE TOMORROW: "+str(len(close_tomorrow)))
    print("TAM CLOSE: "+str(len(file['close'])))
    #for i in range(len(close_tomorrow)):
    #    file['close_tomorrow'] = close_tomorrow

    file = file.assign(close_tomorrow=close_tomorrow.values)
    y = file['close_tomorrow'].values
    file = file.drop(['timestamp', 'volume','dividend_amount', 'adjusted_close','close_tomorrow','split_coefficient'], axis=1)
    dataset = file.values
    #dataset = dataset.astype('float32')
    #print("yMAX: ",max(y))
    #y = y/max(y)

    print(file)
    
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #scaler = scaler.fit(dataset)
    #print('Min: , Max: ', scaler.data_min_, scaler.data_max_)
    # normalize the dataset and print the first 5 rows
    #dataset = scaler.transform(dataset)
    #print(dataset)

    X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.50, random_state=42)

    #MLPRegressor(X_train, X_test, y_train, y_test)
    reg = LinearRegression(X_train, X_test, y_train, y_test)
    valor = np.array([32.34, 32.47, 31.62, 32.32]).reshape(1,-1) 
    print(reg.predict(valor))
    #valor['open'] = 32.34
    #valor['high'] = 32.47
    #valor['low'] = 32.47
    #valor['close'] = 31.62
    #print(reg.predict(valor))


if __name__ == "__main__":
        main()

