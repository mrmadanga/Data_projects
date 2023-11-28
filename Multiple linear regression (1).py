#!/usr/bin/env python
# coding: utf-8

# # Multiple linear regression  on marketing promotion.
# 
# 

# ## Introduction

# Analyze a small business' historical marketing promotion data. Each row corresponds to an independent marketing promotion where the business uses TV, social media, radio, and influencer promotions to increase sales. 
# 
# To address the business' request, we will conduct a multiple linear regression analysis to estimate sales from a combination of independent variables. This will include:
# 
# * Exploring and cleaning data
# * Using plots and descriptive statistics to select the independent variables
# * Creating a fitting multiple linear regression model
# * Checking model assumptions
# * Interpreting model outputs and communicating the results to non-technical stakeholders

# ## Step 1: Imports

# ### Import packages

# Import relevant Python libraries and modules.

# In[1]:


# Import libraries and modules.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols


# ### Load dataset

# In[3]:


# IMPORT DATA.

data = pd.read_csv('marketing_sales_data.csv')

# Display the first five rows.

data.head 


# ## Step 2: Data exploration

# 
# 
# We start with an exploratory data analysis to familiarize ourself with the data and prepare it for modeling.
# 
# The features in the data are:
# 
# * TV promotional budget (in "Low," "Medium," and "High" categories)
# * Social media promotional budget (in millions of dollars)
# * Radio promotional budget (in millions of dollars)
# * Sales (in millions of dollars)
# * Influencer size (in "Mega," "Macro," "Micro," and "Nano" categories)
# 

# The main resons we are performing EDA before constructing a multiple linear regression models include;
# 
# 1.Understanding which variables arepresent in the data.
# 
# 2.Revewing the distribution of features such as minimum, mean,and maximum values.
# 
# 3.Plotting the relationship between the independent and dependent variables to visualizewhich features have alinear relationship.
# 
# 4. identifying issues with the data, such as incorrect values(e.g typos) ormissing values.

# ### Create a pairplot of the data
# 
# Create a pairplot to visualize the relationship between the continous variables in `data`.

# In[4]:


# Create a pairplot of the data.

sns.pairplot(data)


# **Question:** Which variables have a linear relationship with `Sales`? Why are some variables in the data excluded from the preceding plot?
# 
# 

# `Radio` and `Social media` both appear to have linear relationship with `Sales`.
# `Radio` and `Social Media` may be useful as independent variables in a multipe linear regression mdel estimatig  `Sales`.
# 
# `TV` and `Influencer` are excluded from the pairplot because they are not numerical.

# ### Calculate the mean sales for each categorical variable

# There are two categorical variables: `TV` and `Influencer`. To characterize the relationship between the categorical variables and `Sales`, find the mean `Sales` for each category in `TV` and the mean `Sales` for each category in `Influencer`. 

# In[6]:


# Calculate the mean sales for each TV category. 

print(data.groupby('TV')['Sales'].mean())
print('')



# Calculate the mean sales for each Influencer category. 

print(data.groupby('Influencer')['Sales'].mean())
print('')


# **Question:** What do you notice about the categorical variables? Could they be useful predictors of `Sales`?
# 
# 

# The average `Sales` for `High` `TV` promotions is considerably higher than for `Medium` and `Low` `TV` promotions. `TV` may be a strong predictor of `Sales`.
# 
# The categories for `Influencer` have different average `Sales`, but the variation is not substantial. `Influencer` may be a weak predictor of `Sales`.
# 
# These results can be investigated further when fitting the multiple linear regression model. 

# ### Remove missing data
# 
# This dataset contains rows with missing values. To correct this, drop all rows that contain missing data.

# In[7]:


# Drop rows that contain missing data and update the DataFrame.

data = data.dropna(axis=0)


# ### Clean column names

# In[8]:


# Rename all columns in data that contain a space. 

data = data.rename(columns={'Social Media' : 'Social_Media'})


# ## Step 3: Model building

# ### Fit a multiple linear regression model that predicts sales
# 
# Using the independent variables of your choice, fit a multiple linear regression model that predicts `Sales` using two or more independent variables from `data`.

# In[9]:


# Define the OLS formula.

ols_formula = 'Sales ~ C(TV) + Radio'


# Create an OLS model.

OLS =ols(formula = ols_formula, data = data)


# Fit the model.

model = OLS.fit()


# Save the results summary.

model_results = model.summary()


# Display the model results.

model_results


# **Question:** Which independent variables did you choose for the model, and why?
# 
# 

# * `TV` was selected, as the preceding analysis showed a strong relationship between the `TV` promotional budget and the average `Sales`.
# * `Radio` was selected because the pairplot showed a strong linear relationship between `Radio` and `Sales`.
# * `Social Media` was not selected because it did not increase model performance and it was later determined to be correlated with another independent variable: `Radio`.
# * `Influencer` was not selected because it did not show a strong relationship to `Sales` in the preceding analysis.

# ### Check model assumptions

# ### Model assumption: Linearity

# Create scatterplots comparing the continuous independent variable(s) you selected previously with `Sales` to check the linearity assumption.

# In[11]:


# Create a scatterplot for each independent variable and the dependent variable.

#create a 1x2 plot figure.
fig,axes = plt.subplots(1, 2, figsize = (8,4))

#Create a scatterplotbetween radio and sales.
sns.scatterplot(x = data['Radio'], y = data['Sales'], ax=axes[0])

# setthe title for the first plot
axes[0].set_title("Radio and Sales")

#Create a scatterplot between social media and sales
sns.scatterplot(x = data['Social_Media'], y = data['Sales'], ax=axes[1])

# set the title for the second plot.
axes[1].set_title("Social Media and Sales")

# Set the xlabel of the second plot
axes[1].set_xlabel("Social Media")

#Use matplotlib's tight_layout() function to add spaces between plots for a cleaner appearance

plt.tight_layout()


# **Question:** Is the linearity assumption met?
# 

# The linearity assumption holds for `Radio`, as there is a clear linear relationship in the scatterplot between `Radio` and `Sales`. `Social Media` was not included in the preceding multiple linear regression model, but it does appear to have a linear relationship with `Sales`.

# ### Model assumption: Independence

# The **independent observation assumption** states that each observation in the dataset is independent. As each marketing promotion (i.e., row) is independent from one another, the independence assumption is not violated.

# ### Model assumption: Normality

# Create the following plots to check the **normality assumption**:
# 
# * **Plot 1**: Histogram of the residuals
# * **Plot 2**: Q-Q plot of the residuals

# In[13]:


# Calculate the residuals.

residuals = model.resid

# Create a 1x2 plot figure
fig, axes = plt.subplots(1, 2, figsize = (8,4))


# Create a histogram with the residuals. 

sns.histplot(residuals, ax=axes[0])

#set the x label of the residual plot.
axes[0].set_xlabel("Residual Value")

#Set the title of the residual plot.
axes[0].set_title("Histogram of Residal")

# Create a Q-Q plot of the residuals.
sm.qqplot(residuals, line='s',ax = axes[1])

#Set the title of the Q-Q plot.
axes[1].set_title("Normal qq plot")

#Use matplotlib's tight_layout() function to dd space between plts for a neate appearance

plt.tight_layout()

# show the plot.
plt.show()


# **Question:** Is the normality assumption met?
# 
# 

# The histogram of the residuals are approximately normally distributed, which supports that the normality assumption is met for this model. The residuals in the Q-Q plot form a straight line, further supporting that this assumption is met.

# ### Model assumption: Constant variance

# Check that the **constant variance assumption** is not violated by creating a scatterplot with the fitted values and residuals. Add a line at $y = 0$ to visualize the variance of residuals above and below $y = 0$.

# In[14]:


# Create a scatterplot with the fitted values from the model and the residuals.

fig = sns.scatterplot(x = model.fittedvalues, y = model.resid)

#set the x axis label
fig.set_xlabel("Fitted Values")

# Set the title.
fig.set_title("Fitted Values v. Residuals")


# Add a line at y = 0 to visualize the variance of residuals above and below 0.

fig.axhline(0)

#Show the plot.
plt.show()


# **Question:** Is the constant variance assumption met?
# 
# 
# 

# The fitted values are in three groups because the categorical variable is dominating in this model, meaning that TV is the biggest factor that decides the sales.
# 
# However, the variance where there are fitted values is similarly distributed, validating that the assumption is met.
# 

# ### Model assumption: No multicollinearity

# The **no multicollinearity assumption** states that no two independent variables ($X_i$ and $X_j$) can be highly correlated with each other. 
# 
# Two common ways to check for multicollinearity are to:
# 
# * Create scatterplots to show the relationship between pairs of independent variables
# * Use the variance inflation factor to detect multicollinearity
# 

# In[15]:


# Create a pairplot of the data.

sns.pairplot(data)


# **Question 8:** Is the no multicollinearity assumption met?
# 
# 

# The preceding model only has one continous independent variable, meaning there are no multicollinearity issues. 
# 
# If a model used both `Radio` and `Social_Media` as predictors, there would be a moderate linear relationship between `Radio` and `Social_Media` that violates the multicollinearity assumption. Furthermore, the variance inflation factor when both `Radio` and `Social_Media` are included in the model is 5.17 for each variable, indicating high multicollinearity.

# ## Step 4: Results and evaluation

# ### Display the OLS regression results
# 
# If the model assumptions are met, you can interpret the model results accurately.
# 
# First, display the OLS regression results.

# In[17]:


# Display the model results summary.

model_results


# **Question:** What is your interpretation of the model's R-squared?
# 

# Using `TV` and `Radio` as the independent variables results in a multiple linear regression model with $R^{2} = 0.904$. In other words, the model explains $90.4\%$ of the variation in `Sales`. This makes the model an excellent predictor of `Sales`. 

# ### Interpret model coefficients

# With the model fit evaluated, we can look at the coefficient estimates and the uncertainty of these estimates.
# 
# Again, display the OLS regression results.

# In[18]:


# Display the model results summary.

model_results


# **Question:** What are the model coefficients?
# 
# 

# When `TV` and `Radio` are used to predict `Sales`, the model coefficients are:
# 
# * $\beta_{0} =  218.5261$
# * $\beta_{TVLow}= -154.2971$
# * $\beta_{TVMedium} = -75.3120$
# * $\beta_{Radio} =  2.9669$

#  We will write the relationship between `Sales` and the independent variables as a linear equation as follows;
# 
# 

# $\text{Sales} = \beta_{0} + \beta_{1}*X_{1}+ \beta_{2}*X_{2}+ \beta_{3}*X_{3}$
# 
# $\text{Sales} = \beta_{0} + \beta_{TVLow}*X_{TVLow}+ \beta_{TVMedium}*X_{TVMedium}+ \beta_{Radio}*X_{Radio}$
# 
# $\text{Sales} = 218.5261 - 154.2971*X_{TVLow} - 75.3120*X_{TVMedium}+ 2.9669 *X_{Radio}$

# **Question:** What is your intepretation of the coefficient estimates? Are the coefficients statistically significant?
# 
# 

# The default `TV` category for the model is `High` since there are coefficients for the other two `TV` categories, `Medium` and `Low`. Because the coefficients for the `Medium` and `Low` `TV` categories are negative, that means the average of sales is lower for `Medium` or `Low` `TV` categories compared to the `High` `TV` category when `Radio` is at the same level.
# 
# For example, the model predicts that a `Low` `TV` promotion is 154.2971 lower on average compared to a `high` `TV` promotion given the same `Radio` promotion.
# 
# The coefficient for `Radio` is positive, confirming the positive linear relationship shown earlier during the exploratory data analysis.
# 
# The p-value for all coefficients is $0.000$, meaning all coefficients are statistically significant at $p=0.05$. The 95% confidence intervals for each coefficient should be reported when presenting results to stakeholders. 
# 
# For example, there is a $95\%$ chance that the interval $[-163.979,-144.616]$ contains the true parameter of the slope of $\beta_{TVLow}$, which is the estimated difference in promotion sales when a `Low` `TV` promotion is chosen instead of a `High` `TV` promotion.
# 

# **Question:** Why is it important to interpret the beta coefficients?
# 
# 

# Beta coefficients allow you to estimate the magnitude and direction (positive or negative) of the effect of each independent variable on the dependent variable. The coefficient estimates can be converted to explainable insights, such as the connection between an increase in TV promotional budgets and sales mentioned previously.

# **Question:** What are you interested in exploring based on your model?
# 
# 

# Potential areas to explore include:
# 
# * Providing the business with the estimated sales given different TV promotions and radio budgets
# * Additional plots to help convey the results, such as using the `seaborn` `regplot()` to plot the data with a best fit regression line

# **Question:** Do you think your model could be improved? Why or why not? How?

# Given how accurate `TV` was as a predictor, the model could be improved by getting a more granular view of the `TV` promotions, such as by considering more categories or the actual `TV` promotional budgets.
# 
# Furthermore, additional variables, such as the location of the marketing campaign or the time of year, could increase model accuracy. 
# 

# ## Conclusion

# 
# 
# **Key Findings**
# According to the model, high TV promotional budgets result in significantly more sales than medium and low TV promotional budgets. For example, the model predicts that a `Low` `TV` promotion is 154.2971 lower on average than a `high` `TV` promotion given the same `Radio` promotion.
# 
# The coefficient for radio is positive, confirming the positive linear relationship shown earlier during the exploratory data analysis.
# 
# The p-value for all coefficients is $0.000$, meaning all coefficients are statistically significant at $p=0.05$. The 95% confidence intervals for each coefficient should be reported when presenting results to stakeholders. 
# 
# For example, there is a $95\%$ chance the interval $[-163.979,-144.616]$ contains the true parameter of the slope of $\beta_{TVLow}$, which is the estimated difference in promotion sales when a low TV promotional budget is chosen instead of a high TV promotion budget.
# 
# 
# 
# **Findings to external stakeholders**
# High TV promotional budgets have a substantial positive influence on sales. The model estimates that switching from a high to medium TV promotional budget reduces sales by $\$75.3120$ million (95% CI $[-82.431,-68.193])$, and switching from a high to low TV promotional budget reduces sales by $\$154.297$ million (95% CI $[-163.979,-144.616])$. The model also estimates that an increase of $\$1$ million in the radio promotional budget will yield a $\$2.9669$ million increase in sales (95% CI $[2.551,3.383]$).
# 
# Thus, it is recommended that the business allot a high promotional budget to TV when possible and invest in radio promotions to increase sales. 
# 
# 
# 

# ** Compiled by Henry Madanga.**

# In[ ]:




