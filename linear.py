import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


file = pd.read_csv('input/daily_adjusted_EWZ.csv')

#print(file)

y = file['close'].values
file = file.drop(['timestamp','volume','dividend_amount', 'close', 'adjusted_close','split_coefficient'], axis=1)
print(file)
dataset = file.values

dataset = dataset.astype('float32')


X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.20)

print(dataset)

reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_pred, y_test))
#resp = reg.score(X_test,y_test)
#plt.plot(y_test)
plt.plot(y_pred-y_test, color='red')
plt.show()

#print(resp)
