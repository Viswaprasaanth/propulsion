#LINEAR SUPPORT VECOR REGRESSOR FOR GT Turbine decay state coefficient
from sklearn.svm import SVR
regressor=SVR(kernel='linear',degree=1)
from sklearn.model_selection import train_test_split
regressor.fit(X_train,y2_train)
pred=regressor.predict(X_test)
print(regressor.score(X_test,pred))
mse=np.square(np.subtract(y2_test,pred)).mean()
print ("MSE: ",mse)