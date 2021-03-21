# Univariate Analysis
As mentioned in my previous article [Introduction to Univariate, Bivariate and Multivariate Analysis](https://medium.com/analytics-vidhya/univariate-bivariate-and-multivariate-analysis-8b4fc3d8202c), this article will dive a bit deeper into the different analysis. We will use a Kaggle dataset ([Rain in Australia](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package?select=weatherAUS.csv) to conduct our analysis. 
### Univariate recap
Univariate analysis analyzes  only one variable. The most common methods to conduct univariate analysis is to check for central tendency numerical variables and frequency distribution for categorical variables. 
### To get started
- download the dataset
- import necessary packages
- read in the file


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# changing display options to increase the number of columns and rows viewable
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 200)
```


```python
df = pd.read_csv('weatherAUS.csv')
df.head()
```




![png](Intro%20to%20Univariate%20Analysis_files/output_2_0.png)



Using .info() method, we can see the counts and the datatypes of each fature.

This dataset consiist of 145,460 observations (rows) and 23 features (columns).


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 145460 entries, 0 to 145459
    Data columns (total 23 columns):
    Date             145460 non-null object
    Location         145460 non-null object
    MinTemp          143975 non-null float64
    MaxTemp          144199 non-null float64
    Rainfall         142199 non-null float64
    Evaporation      82670 non-null float64
    Sunshine         75625 non-null float64
    WindGustDir      135134 non-null object
    WindGustSpeed    135197 non-null float64
    WindDir9am       134894 non-null object
    WindDir3pm       141232 non-null object
    WindSpeed9am     143693 non-null float64
    WindSpeed3pm     142398 non-null float64
    Humidity9am      142806 non-null float64
    Humidity3pm      140953 non-null float64
    Pressure9am      130395 non-null float64
    Pressure3pm      130432 non-null float64
    Cloud9am         89572 non-null float64
    Cloud3pm         86102 non-null float64
    Temp9am          143693 non-null float64
    Temp3pm          141851 non-null float64
    RainToday        142199 non-null object
    RainTomorrow     142193 non-null object
    dtypes: float64(16), object(7)
    memory usage: 25.5+ MB


We see that the date column and location column have all 145,460 observations but rest of the features are missing observations. Depending on the goal of the analysis and the domaine knowledge you possess, you will have to deal with the null values. For the purpose of this article, I will convert the null values to 0.

A quiick way to get a glimps of the dataset is to use the [.describe()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) method in pandas.

This method provides a summary on all the numerical data within the dataset. 
As you can see below, it shows the count, mean, standard deviation, minimum value, 25%, 50% and 75% percentile and lastly the maxium value of each feature. 


```python
df.describe() #before replacing all null values to 0
```




![png](Intro%20to%20Univariate%20Analysis_files/output_7_0.png)




```python
df.fillna(0, inplace = True) 
# inplace = True replaces all null values in the same dataset.
# by default, to prevent errors, pandas will create a copy of the dataset where 
# all null values are filled if inplace=True is not speficied.

```

Now lets check for null values again. 


```python
df.info()
# there are no more null values in the dataframe
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 145460 entries, 0 to 145459
    Data columns (total 23 columns):
    Date             145460 non-null object
    Location         145460 non-null object
    MinTemp          145460 non-null float64
    MaxTemp          145460 non-null float64
    Rainfall         145460 non-null float64
    Evaporation      145460 non-null float64
    Sunshine         145460 non-null float64
    WindGustDir      145460 non-null object
    WindGustSpeed    145460 non-null float64
    WindDir9am       145460 non-null object
    WindDir3pm       145460 non-null object
    WindSpeed9am     145460 non-null float64
    WindSpeed3pm     145460 non-null float64
    Humidity9am      145460 non-null float64
    Humidity3pm      145460 non-null float64
    Pressure9am      145460 non-null float64
    Pressure3pm      145460 non-null float64
    Cloud9am         145460 non-null float64
    Cloud3pm         145460 non-null float64
    Temp9am          145460 non-null float64
    Temp3pm          145460 non-null float64
    RainToday        145460 non-null object
    RainTomorrow     145460 non-null object
    dtypes: float64(16), object(7)
    memory usage: 25.5+ MB


Lets say I am only interested in Albury and want to know the minimum and maximum temperature there. 
First I would need to isolate all observations for Albury and then do my analysis. 


```python
# this function checks for "Albury" in the "Location" column and returns a boolean. 
df["Location"]=="Albury" 
# this function returns a dataframe where the the "Location" is "Albury". 
df[df["Location"]=="Albury"]
```




![png](Intro%20to%20Univariate%20Analysis_files/output_12_0.png)



I will now assign a variable named albury_df to be the dataframe that consist only of Albury city 


```python
albury_df = df[df["Location"]=="Albury"]
```

As mentioned before, we are only interested in the lowest and the highest temperature recorded in Albury. 


```python
lowest = albury_df["MinTemp"].min()
highest = albury_df["MaxTemp"].max()

print(lowest, highest)
```

    -2.8 44.8


But you could also use the .describe() method to see a lot more information on the features.

You can see the lowest and highest values for MinTemp and MaxTemp. 


```python
albury_df.describe()
```




![png](Intro%20to%20Univariate%20Analysis_files/output_18_0.png)



However, if you are only interested in seeing the spread or frequency of the MinTemp, you could use a bargraph, boxplots, or violinplot or a distribution plot. I personally favor violinplot, but it is not used widely. I wrote an article on [why violinplot should be used more](https://medium.com/analytics-vidhya/a-violin-is-better-f7068129a14)


```python
sns.violinplot(x="MinTemp", data=albury_df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fd48b157278>




![png](Intro%20to%20Univariate%20Analysis_files/output_20_1.png)



```python
sns.distplot(albury_df["MinTemp"])
# here you can see the distribution of the MinTemp.
# y-axis represents the percentage 
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fd48ba71668>




![png](Intro%20to%20Univariate%20Analysis_files/output_21_1.png)


Now lets take a look at a categorical variable. We will look at the WindGustDir. Here we will use the Countplot, since there are a limited number of directions. 


```python
fig, ax = plt.subplots(figsize = (10,8))
sns.countplot(x="WindGustDir", data=albury_df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fd4997f0be0>




![png](Intro%20to%20Univariate%20Analysis_files/output_23_1.png)


There are ways to prettify this to make it more presentable. The starter code can be found on [this article](https://dorjeys3.medium.com/matplotlib-quick-and-pretty-enough-to-get-you-started-5f425b55132f).

Another method to simply see the distribution in numbers would be to create a pivot table in pandas. 



```python
duplicates = albury_df.pivot_table(index= ["WindGustDir"], aggfunc="size")
print(duplicates)
```

    WindGustDir
    0       32
    E      122
    ENE    149
    ESE    113
    N      165
    NE     157
    NNE    151
    NNW    152
    NW     166
    S       60
    SE     279
    SSE    206
    SSW     55
    SW     120
    W      505
    WNW    375
    WSW    233
    dtype: int64


As mentioned in the [Introduction to Univariate, Bivariate and Multivariate Analysis](https://medium.com/analytics-vidhya/univariate-bivariate-and-multivariate-analysis-8b4fc3d8202c), the Univariate analysis is the simplest of the three. It provides basic information but crucial information on one feature. Just by looking at the above table and figure, we are certain that majority of the gust is coming from the Westward direction. We also know the lowest and hightest temperature recorded in the city of Albury. 

In the next article, we will be looking at Bivariate Analysis and diving deeper into the different methods that can be used to conduct the analysis. 


```python

```
