# Linear_Regression
Explanation of Linear Regression using  Ecommerce Dataset
![WhatsApp Image 2021-05-17 at 11 46 07 PM](https://user-images.githubusercontent.com/82372055/118537119-2def3580-b76a-11eb-9b98-c7074a292184.jpeg)

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
![12341234123](https://user-images.githubusercontent.com/82372055/118537954-32681e00-b76b-11eb-8667-6e9186f86fb6.png)

## Exploratory Data Analysis:
Exploratory data analysis is an approach to analyzing data sets to summarize their main characteristics, often using statistical graphics and other data visualization methods
```python
sns.heatmap(df.corr(),annot=True)
```
Using this we get correlation between data  
![Screenshot 2021-05-17 235512](https://user-images.githubusercontent.com/82372055/118538067-575c9100-b76b-11eb-907b-2cd21577d818.png)
```python
sns.jointplot(x='Time on Website', y='Yearly Amount Spent',data=df)
```
Using this we get jointplot between website and Yearly amount
![Screenshot 2021-05-17 235546](https://user-images.githubusercontent.com/82372055/118538119-6b07f780-b76b-11eb-957e-91a755690975.png)
```python
sns.jointplot(x='Time on App', y='Yearly Amount Spent',data=df)
```
Using this we get joinplot between Time on app and Yearly Amount Spent
![Screenshot 2021-05-17 235637](https://user-images.githubusercontent.com/82372055/118538225-88d55c80-b76b-11eb-9348-3d4b985530de.png)
```python
sns.pairplot(df) 
```
this plots multiple plots considering any two data from dataset's to all data
We could see a linear relation between 'Length of Membership' and 'Yearly Amount Spent' also between 'Time on App' and 'Yearly Amount Spent'.

![download](https://user-images.githubusercontent.com/82372055/118538354-b1f5ed00-b76b-11eb-9f8d-32802fb6e8a4.png)


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
```python
plt.scatter(y_test, predictions)
plt.xlabel('Original')
plt.ylabel('Predictions')
```
![Screenshot 2021-05-17 235844](https://user-images.githubusercontent.com/82372055/118538521-ed90b700-b76b-11eb-89b3-2a43773b57f9.png)

now lets check the mean_absolute error,mean_squared_error,root_mean_squared_error

to get the errors we need to import metrics from sklearn  
```python 
from sklearn import metrics
```
```python
print(metrics.mean_absolute_error(y_test, predictions))
print(metrics.mean_squared_error(y_test, predictions))
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```
#### 7.228148653430845
#### 79.81305165097463
#### 8.933815066978644

the above values are the mean_absolute error,mean_squared_error,root_mean_squared_errors respectively

now lets check the coefficients of the data to get an solution for the given dataset 
```python
coef = pd.DataFrame(lm.coef_, X.columns)
coef.columns = ['Coef']
coef
```
![image](https://user-images.githubusercontent.com/82372055/118539367-f9c94400-b76c-11eb-9f40-b1c1d9e6ca22.png)

### so we can see the users are spending more time on App so the the company must look after the contribution of spending more time on making website user friendly to increase the users
