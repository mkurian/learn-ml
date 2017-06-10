# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

from pandas import DataFrame, read_csv

import matplotlib.pyplot as plt
import sys #only needed to determine Python version number
import matplotlib #only needed to determine Matplotlib version number

# Enable inline plotting
#matplotlib inline

print('Python version '+ sys.version)
print('Pandas version ' + pd.__version__)
print('Matplotlib version ' + matplotlib.__version__)

## Lesson 1 ##

## create dataset -- baby names
#-------------
names = ['Bob' , 'Jessica', 'Mary', 'John', 'Mel']
births = [968, 155, 77, 578, 973]
BabyDataSet = list(zip(names,births))
df = pd.DataFrame(data = BabyDataSet, columns = ['Names', 'Births'])

location = '/Users/mkurian/Git/learn-ml/births1880.csv'
df.to_csv(location, index=False, header=True)
df_read = pd.read_csv(location)

#prepare data
#-------------
# Check data type of the columns
df.dtypes
# Check data type of Births column
df.Births.dtype

# analyze data 
#-------------
#find most popular baby name

#method 1
Sorted = df.sort_values(['Births'], ascending = False)
Sorted.head(1)

#method 2
df['Births'].max()

#present data
#-------------

df['Births'].plot()
MaxValue = df['Births'].max()
MaxName = df['Names'][df['Births'] == MaxValue].values

Text = str(MaxValue) + " - " + MaxName

plt.annotate(Text, xy=(1, MaxValue), xytext=(8,0), xycoords = ('axes fraction', 'data'),
            textcoords = 'offset points')
print("Most popular name")
df[df['Births'] == MaxValue]



## Lesson 2 ##

#create data
#----------
from numpy import random
import matplotlib.pyplot as plt

# The inital set of baby names
names = ['Bob','Jessica','Mary','John','Mel']

random.seed(500)
random_names = [names[random.randint(low=0, high=len(names))] for i in range(1000)]
random_names[:10]

births = [random.randint(low=0,high=1000) for i in range(1000)]
births[:10]

BabyDataSet = list(zip(random_names, births))
BabyDataSet[:10]

df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])
df[:10]
df.info()

df.head()

df.tail()

#prepare data

# Method 1:
#get count by name

df['Names'].unique()
# If you actually want to print the unique values:
for x in df['Names'].unique():
    print(x)
    
# Method 2:
print(df['Names'].describe())   

name_aggr = df.groupby('Names')
df = name_aggr.sum()
df

#analyze data 
#find most popular name
# Method 1:
Sorted = df.sort_values(['Births'], ascending=False)
Sorted.head(1)

# Method 2:
df['Births'].max()



#set seed
random.seed(111)

#function to generate test data
def CreateDataSet(Number=1):
    Output = []
    for i in range(Number):
            #create a weekly(mondays) date range
            rng = pd.date_range(start='1/1/2009', end ='12/31/2012',freq='W-MON')
            
            #create random data
            data = random.randint(low=25, high=1000, size=len(rng))
            #status pool
            status = [1,2,3]
            #make random list of statuses
            random_status = [status[random.randint(low=0,high=len(status))] for i in range(len(rng))]
            
            #state pool
            states = ['GA','FL','fl','NY','NJ','TX']
            #make random list of states
            random_states = [states[random.randint(low=0,high=len(states))] for i in range(len(rng))]
            
            Output.extend(zip(random_states, random_status, data, rng))
    return Output

dataset = CreateDataSet(4)
df = pd.DataFrame(data=dataset, columns=['State', 'Status', 'CustomerCount', 'StatusDate'])
df.info()
df.head()

df.to_excel('Lesson3.xlsx', index=False)
# Parse a specific sheet
df = pd.read_excel('Lesson3.xlsx', 0, index_col='StatusDate')
df.dtypes

# Prepare Data
# This section attempts to clean up the data for analysis.

# Make sure the state column is all in upper case
df['State'].unique()
df['State'] = df.State.apply(lambda x:x.upper())
df['State'].unique()

# Only select records where the account status is equal to "1"
mask = df['Status'] == 1
df = df[mask]

# Merge (NJ and NY) to NY in the state column
df['State'][df['State'] == 'NJ'] = 'NY'

# # Convert NJ to NY
# mask = df.State == 'NJ'
# df['State'][mask] = 'NY'

# Remove any outliers (any odd results in the data set)
# At this point we may want to graph the data to check for any outliers or inconsistencies in the data. We will be using the plot() attribute of the dataframe.
# As you can see from the graph below it is not very conclusive and is probably a sign that we need to perform some more data preparation.

df['CustomerCount'].plot(figsize=(15,5))
# Group by State and StatusDate
Daily = df.reset_index().groupby(['State','StatusDate']).sum()
Daily.head()



# What is the index of the dataframe
Daily.index

# Select the State index
Daily.index.levels[0]

# Select the StatusDate index
Daily.index.levels[1]


#Lets now plot the data per State.

#As you can see by breaking the graph up by the State column we have a much clearer picture on how the data looks like. Can you spot any outliers?
Daily.loc['FL'].plot()
Daily.loc['GA'].plot()
Daily.loc['NY'].plot()
Daily.loc['TX'].plot();


Daily.loc['FL']['2012':].plot()
Daily.loc['GA']['2012':].plot()
Daily.loc['NY']['2012':].plot()
Daily.loc['TX']['2012':].plot();

# We will assume that per month the customer count should remain relatively steady. Any data outside a specific range in that month will be removed from the data set. The final result should have smooth graphs with no spikes.

# StateYearMonth - Here we group by State, Year of StatusDate, and Month of StatusDate.
# Daily['Outlier'] - A boolean (True or False) value letting us know if the value in the CustomerCount column is ouside the acceptable range.

# We will be using the attribute transform instead of apply. The reason is that transform will keep the shape(# of rows and columns) of the dataframe the same and apply will not. By looking at the previous graphs, we can realize they are not resembling a gaussian distribution, this means we cannot use summary statistics like the mean and stDev. We use percentiles instead. Note that we run the risk of eliminating good data.

# Calculate Outliers
StateYearMonth = Daily.groupby([Daily.index.get_level_values(0), Daily.index.get_level_values(1).year, Daily.index.get_level_values(1).month])
Daily['Lower'] = StateYearMonth['CustomerCount'].transform( lambda x: x.quantile(q=.25) - (1.5*x.quantile(q=.75)-x.quantile(q=.25)) )
Daily['Upper'] = StateYearMonth['CustomerCount'].transform( lambda x: x.quantile(q=.75) + (1.5*x.quantile(q=.75)-x.quantile(q=.25)) )
Daily['Outlier'] = (Daily['CustomerCount'] < Daily['Lower']) | (Daily['CustomerCount'] > Daily['Upper']) 

# Remove Outliers
Daily = Daily[Daily['Outlier'] == False]

# The dataframe named Daily will hold customer counts that have been aggregated per day. The original data (df) has multiple records per day. We are left with a data set that is indexed by both the state and the StatusDate. The Outlier column should be equal to False signifying that the record is not an outlier.
Daily.head()

# We create a separate dataframe named ALL which groups the Daily dataframe by StatusDate. We are essentially getting rid of the State column. The Max column represents the maximum customer count per month. The Max column is used to smooth out the graph.
# Combine all markets

# Get the max customer count by Date
ALL = pd.DataFrame(Daily['CustomerCount'].groupby(Daily.index.get_level_values(1)).sum())
ALL.columns = ['CustomerCount'] # rename column

# Group by Year and Month
YearMonth = ALL.groupby([lambda x: x.year, lambda x: x.month])

# What is the max customer count per Year and Month
ALL['Max'] = YearMonth['CustomerCount'].transform(lambda x: x.max())
ALL.head()

# There is also an interest to gauge if the current customer counts were reaching certain goals the company had established. The task here is to visually show if the current customer counts are meeting the goals listed below. We will call the goals BHAG (Big Hairy Annual Goal).
# Create the BHAG dataframe
data = [1000,2000,3000]
idx = pd.date_range(start='12/31/2011', end='12/31/2013', freq='A')
BHAG = pd.DataFrame(data, index=idx, columns=['BHAG'])
BHAG

# Combine the BHAG and the ALL data set 
combined = pd.concat([ALL,BHAG], axis=0)
combined = combined.sort_index(axis=0)
combined.tail()

fig, axes = plt.subplots(figsize=(12, 7))

combined['BHAG'].fillna(method='pad').plot(color='green', label='BHAG')
combined['Max'].plot(color='blue', label='All Markets')
plt.legend(loc='best');

# There was also a need to forecast next year's customer count and we can do this in a couple of simple steps. We will first group the combined dataframe by Year and place the maximum customer count for that year. This will give us one row per Year.
# Group by Year and then get the max value per year
Year = combined.groupby(lambda x: x.year).max()
Year

# Add a column representing the percent change per year
Year['YR_PCT_Change'] = Year['Max'].pct_change(periods=1)
Year

# To get next year's end customer count we will assume our current growth rate remains constant. We then will increase this years customer count by that amount and that will be our forecast for next year.
(1 + Year.ix[2012,'YR_PCT_Change']) * Year.ix[2012,'Max']

#present data
#individual grpahs per state

# First Graph
ALL['Max'].plot(figsize=(10, 5));plt.title('ALL Markets')

# Last four Graphs
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
fig.subplots_adjust(hspace=1.0) ## Create space between plots

Daily.loc['FL']['CustomerCount']['2012':].fillna(method='pad').plot(ax=axes[0,0])
Daily.loc['GA']['CustomerCount']['2012':].fillna(method='pad').plot(ax=axes[0,1]) 
Daily.loc['TX']['CustomerCount']['2012':].fillna(method='pad').plot(ax=axes[1,0]) 
Daily.loc['NY']['CustomerCount']['2012':].fillna(method='pad').plot(ax=axes[1,1]) 

# Add titles
axes[0,0].set_title('Florida')
axes[0,1].set_title('Georgia')
axes[1,0].set_title('Texas')
axes[1,1].set_title('North East');


