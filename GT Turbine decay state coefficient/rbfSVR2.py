#rbf SUPPORT VECTOR REGRESSOR for GT Turbine decay state coefficient
from sklearn.metrics import r2_score
regressor=SVR(kernel='rbf',epsilon=1.0)
regressor.fit(X_train,y2_train)
pred=regressor.predict(X_test)
print(regressor.score(X_test,pred))
mse=np.square(np.subtract(y2_test,pred)).mean()
print ("MSE: ",mse)