# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.
   

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MURALIDHARAN M
RegisterNumber:  212223040120
*/

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
df.head()
print(df.head())
df.tail()
print(df.tail())
X=df.iloc[:,:-1].values
X
print(X)
Y=df.iloc[:,1].values
Y
print(Y)

#spilitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred
print(Y_pred)
Y_test
print(Y_test)

#graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)


## Output:
![1](https://github.com/user-attachments/assets/499b83f5-4780-4adc-a06f-697dbd46ae9a)

![2](https://github.com/user-attachments/assets/71c6d8cc-d8e6-4077-8b86-a7b8b92d59f4)

![3](https://github.com/user-attachments/assets/406a5535-753e-4285-8355-ef96663fe0a0)

![4](https://github.com/user-attachments/assets/07845d15-9f2b-481b-a71c-5b28377c5c48)

![5](https://github.com/user-attachments/assets/6837eb73-4c34-4af2-973b-b04d8ff7b01b)

![6](https://github.com/user-attachments/assets/1a3d9113-05c0-4864-a9de-b3ad6d504f0b)

![7](https://github.com/user-attachments/assets/7ea8965c-c753-4518-8e61-cc3ac4e2a6bc)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
