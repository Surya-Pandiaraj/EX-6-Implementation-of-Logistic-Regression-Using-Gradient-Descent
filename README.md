### NAME: SURYA P <br>
### REG NO: 212224230280

# EX-6-Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM :

To write a program to implement the the Logistic Regression Using Gradient Descent.

## EQUIPMENTS REQUIRED :

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook


## ALGORITHM :
 
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for sigmoid, loss, gradient and predict and perform operations.


## PROGRAM :

```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Surya P
RegisterNumber:  212224230280
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("C:\\Users\\admin\\Downloads\\Placement_Data (1).csv")

dataset

dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
theta=np.random.randn(X.shape[1])

y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
    
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
    
y_pred = predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)

print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)

print(y_prednew)
```
## OUTPUT :

![image](https://github.com/user-attachments/assets/3ff5abac-1e7b-4ba5-8126-4ef83129c142)

#### Read the file and display:
![image](https://github.com/POZHILANVD/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870498/cb244b14-56e7-491b-a316-3f7f6821e1a7)
#### Categorizing columns:
![image](https://github.com/POZHILANVD/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870498/1419f9fd-7720-4c06-8e22-d42be9e58378)
#### Labelling columns and displaying dataset:
![image](https://github.com/POZHILANVD/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870498/0a1a2fb4-c8aa-4cf7-90fb-cc7f12b94e58)
#### Display dependent variable:
![image](https://github.com/POZHILANVD/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870498/dcc8c7a9-045a-48db-b0f6-26b8816e7c59)
#### Printing accuracy:
![image](https://github.com/POZHILANVD/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870498/43a1af40-a59b-4ee9-9adc-2fa98b0744b7)
#### Printing Y:
![image](https://github.com/POZHILANVD/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870498/d5812479-a68a-47e0-bd9d-971d61dfc2bd)
#### Printing y_prednew:
![image](https://github.com/POZHILANVD/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144870498/4731cafd-ee77-4464-aa1a-cf03ea783802)


## RESULT :
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
