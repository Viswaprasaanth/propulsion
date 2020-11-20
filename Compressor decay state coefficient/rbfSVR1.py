#rbf SUPPORT VECTOR REGRESSOR for Compressor decay state coefficient
from sklearn.metrics import r2_score
regressor=SVR(kernel='rbf',epsilon=1.0)
regressor.fit(X_train,y1_train)
pred=regressor.predict(X_test)
print(regressor.score(X_test,pred))
mse=np.square(np.subtract(y1_test,pred)).mean()
print ("MSE: ",mse)