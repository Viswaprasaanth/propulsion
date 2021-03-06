#DECISION TREE REGRESSOR FOR GT Compressor decay state coefficient
from sklearn.tree import DecisionTreeRegressor 
regressor = DecisionTreeRegressor(random_state = 0) 
regressor.fit(X_train, y1_train) 
y_pred = regressor.predict(X_test) 
print(regressor.score(X_test,y_pred))
import numpy as np
mse=np.square(np.subtract(y1_test,y_pred)).mean()
print ("MSE: ",mse)