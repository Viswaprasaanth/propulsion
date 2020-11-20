import pandas as pd
data=pd.read_excel("/content/propulsion1.xlsx")
data
data.describe()
data.isnull()
data.boxplot()

#spliting the data 
x= data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]].values 
y1=data["GT Compressor decay state coefficient."]
y2=data["GT Turbine decay state coefficient."]

# split a dataset into train and test sets
from sklearn.model_selection import train_test_split
# split into train test sets
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x, y1, y2, test_size=0.3)
print(X_train.shape, X_test.shape, y1_train.shape, y1_test.shape, y2_train.shape, y2_test.shape)
#feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)