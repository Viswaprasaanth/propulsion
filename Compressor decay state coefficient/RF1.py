#RANDOM FOREST REGRESSOR for GT Compressor decay state coefficient
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train, y1_train)
y_pred = regr.predict(X_test) 
print(regr.score(X_test,y_pred))
mse=np.square(np.subtract(y1_test,y_pred)).mean()
print ("MSE: ",mse)