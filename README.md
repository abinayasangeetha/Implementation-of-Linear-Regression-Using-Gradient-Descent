# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the Dataframe
2. Write a function Compute Cost to generate the cost function
3. Perform iterations of gradient steps with learning rate
4. Plot the cost function using gradient descent and generate the required Graph


## Program:
```
Developed by: ABINAYA S
RegisterNumber:  212222230002
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linearREG(X1,Y,learnRate=0.01,Iteration=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(Iteration):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-Y).reshape(-1,1)
        theta-=learnRate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('/content/50_Startups.csv',header=None)
print(data.head())
X=(data.iloc[1:,:-2].values) # Assuming the last column is your target variable 'Y' and the preceding column
X1=X.astype(float)
scaler=StandardScaler()
Y=(data.iloc[1:,-1].values).reshape(-1,1)
X1scaled=scaler.fit_transform(X1)
Y1scaled=scaler.fit_transform(Y)
theta=linearREG(X1scaled,Y1scaled)
theta=linearREG(X1scaled,Y1scaled)                             # Learn model parameters
newData=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)   # Predict target value for a new data point
newScaled=scaler.fit_transform(newData)
prediction=np.dot(np.append(1, newScaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")



```

## Output:
![Screenshot 2024-03-08 210327](https://github.com/abinayasangeetha/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393675/6db8dff6-a688-4b82-b23a-ca66ad099c5a)

![Screenshot 2024-03-08 210342](https://github.com/abinayasangeetha/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393675/3c537983-8738-481b-9a42-3ebcb5321930)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
