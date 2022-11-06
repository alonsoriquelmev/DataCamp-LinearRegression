import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import classification_report, confusion_matrix
plt.style.use('ggplot')

boston = pd.read_csv('boston.csv')

boston = boston.drop(columns = 'Unnamed: 0')

print(boston.head())


X = np.array(boston.drop('medv', axis = 1))
y = np.array(boston['medv'])

X_rooms = X[:,5]


y = y.reshape(-1,1)
X_rooms = X_rooms.reshape(-1,1)

plt.figure(figsize = (10,7))
plt.scatter(X_rooms,y)
plt.ylabel('Value of house /1000 ($)')
plt.xlabel('Number of rooms')
plt.show()


reg = LinearRegression()
reg.fit(X_rooms, y)

prediction_space = np.linspace(min(X_rooms),
                               max(X_rooms)).reshape(-1,1)

plt.figure()
plt.scatter(X_rooms, y, color = 'blue')
plt.plot(prediction_space, reg.predict(prediction_space), color = 'black', linewidth = 3)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X , y,test_size = 0.3, random_state = 42)
reg_all = LinearRegression()

cv_results = cross_val_score(reg_all, X_train, y_train, cv = 5) #Cross Validation

reg_all.fit(X_train, y_train)
prediction = reg_all.predict(X_test)


#Ridge Regression
ridge = Ridge(alpha = 0.1, normalize = True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print(ridge.score(X_test, y_test))

#Lasso Regression

names = boston.drop('medv', axis = 1).columns
lasso = Lasso(alpha = 0.1)
lasso_coef = lasso.fit(X,y).coef_

plt.figure()
_ = plt.plot(range(len(names)),lasso_coef)
_ = plt.xticks(range(len(names)), names ,rotation = 60)
_ = plt.ylabel('Coefficients')
plt.show()


