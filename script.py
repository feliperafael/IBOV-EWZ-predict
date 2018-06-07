import pandas as pd
from sklearn import linear_model
from sklearn import neural_network
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def MLPRegressor(X_train, X_test, y_train, y_test):
	print("MLPRegressor")
	error = []
	model = []
	for i in range(10):
		model.append(neural_network.MLPRegressor(hidden_layer_sizes=(100, ),max_iter=1000))
		model[i].fit(X_train,y_train)
		y_pred = model[i].predict(X_test)
		error.append(np.sqrt(mean_squared_error(y_pred, y_test)))
		
	print("RMSE: %.2f" % np.mean(error))
	return model[np.argmin(error)]

def LinearRegression(X_train, X_test, y_train, y_test):
	print("LinearRegression")
	reg = linear_model.LinearRegression()
	reg.fit(X_train,y_train)
	y_pred = reg.predict(X_test)
	print("RMSE: %.2f" % np.sqrt(mean_squared_error(y_pred, y_test)))

	return reg

def main():
	file = pd.read_csv('input/daily_adjusted_EWZ.csv')

	y = file['close'].values
	file = file.drop(['timestamp','volume','dividend_amount', 'close', 'adjusted_close','split_coefficient'], axis=1)
	dataset = file.values
	dataset = dataset.astype('float32')

	X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.20)

	MLPRegressor(X_train, X_test, y_train, y_test)
	LinearRegression(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()

