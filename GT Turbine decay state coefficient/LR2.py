#LINEAR REGRESSION for GT Turbine decay state coefficient
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train, y2_train)
pred = regr.predict(X_test)
print(regr.score(X_test,pred))
mse=np.square(np.subtract(y2_test,pred)).mean()
print ("MSE: ",mse)