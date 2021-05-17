# Linear_Regression
Explanation of Linear Regression using  Ecommerce Dataset
# Linear-Regression

Linear regression is used to predict the continuous dependent variable using a given set of independent  

## Using Linear Regression Algorithm on Ecommerce Dataset:

In the beginning, we import all required libraries
 
```python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
Pandas is used for data analysis. The library allows various data manipulation operations such as merging, reshaping, selecting, as well as data cleaning, and data wrangling features

Numpy is used in the industry for array computing

Seaborn is a Python library used for enhanced data visualization

## Importing Dataset

```python
df = pd.read_csv("/content/Ecommerce Customers")
```
importing dataset is done using .read_csv


## Getting More Information
```python
df.head()
df.info()
df.describe()
```
3 screenshots

## Exploratory Data Analysis:
Exploratory data analysis is an approach to analyzing data sets to summarize their main characteristics, often using statistical graphics and other data visualization methods
```python
sns.heatmap(df.corr(),annot=True)
```
Using this we get correlation between data  
```python
sns.jointplot(x='Time on Website', y='Yearly Amount Spent',data=df)
```
Using this we get jointplot between website and Yearly amount   
```python
sns.jointplot(x='Time on App', y='Yearly Amount Spent',data=df)
```
Using this we get joinplot between Time on app and Yearly Amount Spent

```python
sns.pairplot(df) 
```
plots multiple plots considering any two data from dataset's to all data

pic

We could see a linear relation between 'Length of Membership' and 'Yearly Amount Spent' also between 'Time on App' and 'Yearly Amount Spent'.

As there are no missing values we can now proceed to build the model.

## Training LinearRegression Our Model:
first, we import train test split 
```python
from sklearn.model_selection import train_test_split
```
now at we split Features (X) and Labels (y) as below.
```python
X = df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']
```

Then we split into test and train datasets using the following code
```python
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 101)
```

now we are ready to start fitting the Data in Model using LinearRegression
```python
lr = LinearRegression()
lr.fit(x_train,y_train)
```
Training our linear regression model is Done. The complexity of using more than one independent variable to predict a dependent variable by fitting a best linear is done by using lr.fit(x_train,y_train) method.

## Predictions 
We predict the values for our testing set (x_test) and save it in the predictions variable 

```python
predictions = lr.predict(x_test)
```
now we can check the accuracy score to check how good our model has done training.
```python
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
```
