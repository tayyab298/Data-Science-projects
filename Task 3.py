#!/usr/bin/env python
# coding: utf-8

# # Task 3 - Exploratory Data Analysis - Retail

# Author -Tayyab Khan
# 
# Dataset-https://bit.ly/3i4rbWl
# 
# Copyright- Tayyab Khan 2021

# In[12]:


#importing the required libraries

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


data=pd.read_csv('SampleSuperstore.csv')
data.head(10)


# In[14]:


data.info()


# In[15]:


data.describe()


# * Since the mean value of the featues Sales and Profit are greater than their median values it can be concluded that there are potential outleirs present in the distribution.
# * The features such as Discount and Quantity have near normal distribution as the mean and median values are close to each other.

# ### Numerical and Categorical Features:

# In[16]:


data_Num=data.select_dtypes(np.number)
data_Cat=data.select_dtypes(object)
print('The Numerical features : ')
print(data_Num.columns,'\n')
print('The Categorical features : ')
print(data_Cat.columns)


# ### Checking the symmertry of data:
#     

# In[17]:


data.skew()


# * As observed, features Sales and Profit are right skewed.

# ### finding duplicate values:

# In[18]:


# Checking the number of duplicate entries
print('Number of duplicate entries : ',data.duplicated().sum(),'\n')

# dropping the duplicate entries permanently
data.drop_duplicates(inplace=True)
print('Number of duplicates after treatment : ',data.duplicated().sum())


# ### Checking for Null Values:

# In[19]:


count=data.isna().sum()
percent=((count/data.shape[0])*100)
null=pd.DataFrame(pd.concat([count,percent],keys=['Missing values','% of Missing values'],axis=1))
null


# ## Univariate Analysis:

# In[20]:


#dropping Postal code as it not relevant
data_Num=data_Num.drop('Postal Code',axis=1)
data_Num.columns


# In[21]:


#univariate analysis for numerical feautures:
plt.rcParams['figure.figsize']=(12,8)
j=1
sns.set_style('whitegrid')
for i in data_Num:
    plt.subplot(2,2,j)
    sns.distplot(data[i])
    j+=1
plt.show()


# * there is a postive skewness present in data.

# In[22]:


#univariate analysis of categorical features:
for i in data_Cat:
    count=data[i].value_counts()
    percent=data[i].value_counts(normalize=True)*100
    print(pd.DataFrame({'count':count,'Percentage %':percent}),'\n')


# ## Bivariate Analysis:

# In[23]:


#losses made as per region

print('Subsection in Region: ')
print('\n'.join(data['Region'].unique()),'\n')
print('No of values in each sub-section: ')
print(data['Region'].value_counts())


# In[24]:


plt.figure(figsize=(12,8))
sns.barplot(x=data[data['Profit']<=0]['Region'],y=data[data['Profit']<=0]['Profit'],palette='viridis')
plt.xticks(rotation=45,fontsize=15)
plt.xlabel('Region',fontsize=15)
plt.ylabel('Profit',fontsize=15)
plt.title('Loss made as per Regions',fontsize=15)
plt.show()


# *From the above plot it can be concluded that the South Region suffers maximum loss and should be considered for strict monitoring.
# 
# *The East Region incurrs the 2nd highest loss.

# In[25]:


#losses made as per State:
print('Subsection in Region: ')
print('\n'.join(data['State'].unique()),'\n')
print('No of values in each sub-section: ')
print(data['State'].value_counts())


# In[26]:


plt.figure(figsize=(20,12))
sns.barplot(x=data[data['Profit']<=0]['State'],y=data[data['Profit']<=0]['Profit'],palette='viridis')
plt.xticks(rotation=45,fontsize=15)
plt.xlabel('State',fontsize=15)
plt.ylabel('Profit',fontsize=15)
plt.title('Loss made as per States',fontsize=15)
plt.show()


# *The states in the above plot have a profit which is less than or equal to 0.
# 
# *The states have a magnitude of profit on the negative scale.
# 
# *Proper analysis should be done to identify and rectify the reasons for such poor performance.

# In[27]:


#losses made as per shipment Mode:
print('Sub-section in Ship Mode : ')
print('\n'.join(data['Ship Mode'].unique()),'\n')
print('No of values in each sub-section : ')
print(data['Ship Mode'].value_counts())


# In[28]:


plt.figure(figsize=(12,8))
sns.barplot(x=data['Ship Mode'],y=data[data['Profit']<=0]['Profit'],palette='crest')
plt.xticks(rotation=45,fontsize=15)
plt.xlabel('State',fontsize=15)
plt.ylabel('Profit',fontsize=15)
plt.title('Loss made as per Shipment Mode',fontsize=15)
plt.show()


# * The Shipment made on the same day has incurred maximum losses as compared to the other classes of Shipment mode.

# In[29]:


#losses made as per Segment:
print('Sub-sections in Segment : ')
print('\n'.join(data['Segment'].unique()),'\n')
print('No of values in each sub-section : ')
print(data['Segment'].value_counts())


# In[30]:


plt.figure(figsize=(15,10))
sns.barplot(x=data['Segment'],y=data[data['Profit']<=0]['Profit'],palette='magma_r')
plt.xticks(rotation=45,fontsize=15)
plt.xlabel('Ship Mode',fontsize=15)
plt.ylabel('Profit',fontsize=15)
plt.title('Loss made as per Segment',fontsize=15)
plt.show()


# * The Consumer segment has incurred highest loss.

# In[31]:


#losses as per Category and Sub-category:
data.groupby(['Category','Sub-Category'])['Profit'].sum().plot(kind='bar',figsize=(15,10))
plt.xticks(rotation=45,fontsize=15)
plt.xlabel('Category & Sub-Category',fontsize=15)
plt.ylabel('Profit',fontsize=15)
plt.title('Loss made as per Category & Sub-Category',fontsize=15)
plt.show()


# * Bookcases and Tables in the Furnitures category has suffered losses.
# * Storage in the Office Supplies category has suffered losses.

# ## Comparison of Profit/Loss and Sales w.r.t each business indices:
#     

# In[32]:


#Profit/Loss Vs Sales as per Region:
data.groupby(['Region'])['Profit','Sales'].sum().plot(kind='bar',figsize=(15,10))
plt.xticks(rotation=30,fontsize=15)
plt.xlabel('Region',fontsize=15)
plt.ylabel('Profit/Loss vs Sales',fontsize=15)
plt.title('Profit/Loss Vs Sales as per Region',fontsize=15)
plt.show()


# * West region has highest sales and the highest profit.
# * East Region has second highest sales and second highest profit.
# * South region has sales lower than Central region but has a more profit than Central. Therefore monitoring the Central region is recommended as to understand why they are incurring losses.

# In[33]:


# Profit/Loss Vs Sales as per Category

data.groupby(['Category'])['Profit','Sales'].sum().plot(kind='bar',figsize=(15,10),color=['Magenta','green'])
plt.xticks(rotation=30,fontsize=15)
plt.xlabel('Category',fontsize=15)
plt.ylabel('Profit/Loss vs Sales',fontsize=15)
plt.title('Profit/Loss Vs Sales as per Category',fontsize=15)
plt.show()


# * Technology has highest sales as a result we can see that the profit is significantly higher than other categories.
# * Furniture has the lowest sales and the profits are also the lowest.
# * Furniture needs to be monitored.

# In[34]:


#Profit/Loss Vs Sales as per Sub-Category:
data.groupby(['Category','Sub-Category'])['Profit','Sales'].sum().plot(kind='bar',figsize=(15,10),
                                                        color=['magenta','turquoise'])
plt.xticks(rotation=30,fontsize=15)
plt.xlabel('Sub-Category',fontsize=15)
plt.ylabel('Profit/Loss vs Sales',fontsize=15)
plt.title('Profit/Loss Vs Sales as per Sub-Category',fontsize=15)
plt.show()


# * Bookcases and Tables have incurred losses even after siginifant sales.
# * These sub-categories under Furniture need to be monitored.
# * No profit is generated in the sub-category Storage in spite of significnt sales.
# * Sales of Machines are significantly high but this sub-category has not generated good profits. Hence the focus should be put to understand the reasons.

# In[35]:


# Profit/Loss Vs Sales as per Segment:

data.groupby(['Segment'])['Profit','Sales'].sum().plot(kind='bar',figsize=(15,10),
                                                        color=['maroon','silver'])
plt.xticks(rotation=30,fontsize=15)
plt.xlabel('Segment',fontsize=15)
plt.ylabel('Profit/Loss vs Sales',fontsize=15)
plt.title('Profit/Loss Vs Sales as per Segment',fontsize=15)
plt.show()


# * Home Office has lowest sales and lowest profit.
# * It should be focused on.
