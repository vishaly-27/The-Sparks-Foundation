
# coding: utf-8

# # GRIPAUGUST21

# ## Done by : VISHALY B

# # DATA SCIENCE AND BUSINESS ANALYTICS

# ## TASK-1 Prediction using Supervised ML

# # Introduction:
# 
#    The simple linear model is modelled when the model has only two variables one variable is continuous variable (depended variable),independent variable which can be continuous/categorical. The linear regression model is a classical Supervised learning Algorithm. It is a type of regression analysis where the number of independent variables is one and there is a linear relationship between the independent(x) and dependent(y) variable.
# The model is **Y = β0 + β1X + ϵ**. 
# 
# **Y** is the dependent or study variable(Score),
# 
# **x** is the independent cariable(Study hours), 
# 
# **β0** is intercept 
# 
# **β1** is the slope of the line
# 
# **ϵ** is the error term
# Here, the model will be build training data and accuracy is measured using test data. 
#    
# ### Objective:
# 
#    i) To fit the **simple linear model for predicting score based on number hours studied** using supervisied learning algorithm for simple linear regression. To check the accuracy measure of the model.
#    
#    ii) To predict the score of a student if the study time is 9.25 hrs/ day.   

# ##### Importing the required packages 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Importing the dataset
print("Data imported successfully")
df = pd.read_csv("GRIP_SLR_data.csv")
df.head()


# ## Exploratory Data Analysis

# In[4]:


# Dimension of dataset
print("The number of observations and variables of the data: ",df.shape)


# In[5]:


# Checking the datatype and missing values
df.info()


#  No missing values in the data, also scores is the integer data type and Hours is the float data type.

# In[6]:


# Descriptive summary of the dataset
df.describe()


# From the summary 50% of the data lies very close the mean value.

# In[7]:


x = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values


# ##### Scatter plot for independent(Hours) and dependent variable(Scores)

# In[8]:


plt.scatter(x,y)
plt.title("Scatter plot for Hours and Scores")
plt.xlabel("Hours")
plt.ylabel("Scores")


# The Scatter plot shows that the data points are positively correlated that is the study hours increases marks also increases. Hence, there is a positive linear relationship between number hours studied and the scores of the students.

# ##### Heatmap between independent(Hours) and dependent variable(Scores)

# In[9]:


plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True  )
plt.show()


# From the correlation matrix, obtained **correlation of scores and study hours is 0.98** which implies that there exist a **strong positive correlation** between hours of studied and scores of students.

# In[10]:


# Box plot for independent variable
box1=sns.boxplot("Hours",data=df)
box1.set(title='Box plot for hours')


# Box plot helps to indentify outliers,quartiles and skewness of the variable. Here, no outliers and the 50% of the student's study hours lies between the range of 2.7 to 7.4. The variable hours slightly skewed positively. 

# In[11]:


# Box plot for dependent variable

box1=sns.boxplot("Scores",data=df)
box1.set(title='Box plot for scores')


# Here, no outliers and the 50% of the student'scores lies between the range of 30 to 75, maximum mark is approximately 95 and minimum mark is 15. The variable scores skewed positively. 

# ### Creating Train and  Test Data

# In[12]:


X = df.iloc[:, :-1].values  
Y = df.iloc[:, 1].values


# In[13]:


X


# In[14]:


Y


# In[15]:


# Importing the package for splitting and the train and test dataset
from sklearn.model_selection import train_test_split


# In[16]:


from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(x, y, 
                            test_size=0.2, random_state=0) 


# Splitted the data into training and testing sets of 80% and 20% ratio respectively using train_test_split function.

# ## Training the Algorithm

# In[17]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train,Y_train) 

print("Training complete.")

print("Regressor Coefficient(β1): ",regressor.coef_)
print("Intercept (β0): ",regressor.intercept_)


# #### Inference from the fitted model
# The fitted model is **Scores(Y)= 2.018160041434683 + 9.91065648*Hours**.
# 
# The sign of a regression coefficient tells whether there is a positive or negative correlation between
# the independent variable the dependent variable. A positive coefficient indicates that as the value of the
# independent variable increases, the mean of the dependent variable also tends to increase.
# 
# Regression coefficients represent(9.91065648) the mean change in the response variable(Scores)
# for one hour of change in the predictor variable(Study time)
# The equation shows that the coefficient of study hour is 9.91065648.The coefficient
# indicates that for every additional hour of study time we can expect the score of the student 
# increase by an average of (9.91065648) 10 marks. That is, From the data given,we estimate that the score of the student is increased by 10(9.91065648) marks for a increased unit of study hour.
# 

# ### Fitted line of Train Data 

# In[18]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line,'g');
plt.show()


# It can be seen that not all the data points fall exactly on the fitted regression, some of the points are above the line and some are below it but lies very close to the fitted line.

# ### Making Prediction 

# In[19]:


#making prediction
y_pred = regressor.predict(X_test) 


# In[20]:


# Comparing Actual and Predicted values
df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})  
df


# #### Predicting the score of the student if the study hour is 9.25 hours/day

# In[21]:


hours = 9.25
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# Therefore the predicted score of the student if the study hour **9.25 hours/day** is approximately **94%** ( 93.69173248737538).

# ### Accuracy of the Model 

# In[22]:


from sklearn.metrics import r2_score
r=r2_score(Y_test,y_pred)
print("The r2 score of the model is",r)


# The coefficient of deterination(R square) value is  0.94549. That is **94.5% of the total variation of scores(dependent variabel) explained by the study hours (independent Variable)**. The value of R square is closer to one. The model predicting the score with 94.5% accuracy. Therefore, **the model is a  good fit**. 

# ### Evaluating the Model 

# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.
# 
# 

# In[23]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(Y_test, y_pred)) 

