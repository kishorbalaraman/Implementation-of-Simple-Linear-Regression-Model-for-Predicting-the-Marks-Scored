# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Calculate the mean of the X -values and the mean of the Y -values.
3. Find the slope m of the line of best fit using the formula.
   <img width="231" height="100" alt="192078527-b3b5ee3e-992f-46c4-865b-3b7ce4ac54ad" src="https://github.com/user-attachments/assets/99f10441-22da-425d-a0ba-e4b6621b1cc9" />
4.Compute the y -intercept of the line by using the formula:
<img width="148" height="40" alt="192078545-79d70b90-7e9d-4b85-9f8b-9d7548a4c5a4" src="https://github.com/user-attachments/assets/a972c6cb-27dc-4bb3-ba8d-e4d00372cc8d" />
5. Use the slope m and the y -intercept to form the equation of the line. 6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:
```
# Simple Linear Regression Example
# Developed by: KISHOR B
# Register Number: 212225230141
```
```
# Step 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create dataset (Study Time vs Exam Score)
data = {
    "Study_Time": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Exam_Score": [30, 35, 45, 50, 60, 65, 75, 80, 90, 95]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

# Step 3: Define features and target
X = df[["Study_Time"]]
y = df["Exam_Score"]

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Step 5: Create and train model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = regressor.predict(X_test)

# Step 7: Evaluate model
print("\nModel Details:")
print("Intercept:", regressor.intercept_)
print("Coefficient:", regressor.coef_[0])

print("\nPerformance:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 8: Plot graph
plt.scatter(X, y, label="Actual Data")
plt.plot(X, regressor.predict(X), linewidth=2, label="Best Fit Line")
plt.xlabel("Study Time")
plt.ylabel("Exam Score")
plt.title("Linear Regression Model")
plt.legend()
plt.grid()
plt.show()

# Step 9: Custom prediction
time = float(input("Enter study time: "))
predicted_score = regressor.predict([[time]])
print(f"Predicted score for {time} hours = {predicted_score[0]:.2f}")

```
## Output:
<img width="1919" height="1079" alt="Screenshot 2026-04-27 155618" src="https://github.com/user-attachments/assets/ef3c60d7-96a6-4303-8b88-23faf096e29e" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
