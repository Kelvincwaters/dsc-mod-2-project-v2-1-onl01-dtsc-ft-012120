#### This project entails utilizing the kc_house_data set to predict housing prices making use of a multivariate linear regression model. The initial dataset consists well over twenty-thousand rows spanding twenty-one columns at approx: 3.5MB. The price is the dependant variable and is the targeted prediction in this notebook, the features are the independent variables. The columns and descriptions in the dataset are as follows:
* id - a notation for a house
* date - date house was sold
* price - price is prediction target
* bedrooms - number of bedrooms/house
* bathrooms - number of bathrooms/house
* sqft_living - square footage of the home
* sqft_loft - square footage of the lot
* floors - total floors (levels) in the house
* waterfront - house which has a view to a waterfront
* view - has been viewed
* condition - how good the condition is overall(1-5 scale, 5 being excellent)
* grade - overall grade given to the housing unit, based on King County grading system
* sqft_above - square footage of the house apart from the basement
* sqft_basement - square footage of the basement
* yr_built - year house was constructed


#### Exploratory data analysis (EDA) questions addressed in this notebook are as follows:
1. *What independant variables are being considered categorical and why?*
1. *Which, if any, independant variables were considered highly correlated and how was it handled?*
1. *How were possible outliers delt with?*

#### Coefficient findings and Answers:
1. *Finding one, answer*
1. *Finding two, answer*
1. *Finding three, answer*

> This notebook will be following the **OSEMIN** processing model and is documented accordingly

<img src="images/new_osemn.png" width=600>


#### Which independent variables are being considered categorial and why? 
Views, sqft_basement, and waterfront were changed to binary values 1 or 0 as a way to view these values as "has view, has basement, and is waterfront. It's either a yes or no. In real estate the basement is NOT calculated (much like garages here in the South) in the total sqft of the dwelling even if the basement is considered "finished" much like a home theatre or a dedicated kids play area, that this data doesn't indicate. 

```python
# views to binary
data.view.values[data['view'].values > 0] = 1
# sqft_basement to binary
data.sqft_basement.values[data['sqft_basement'].values > 0] = 1
# waterfront to binary
data.waterfront.values[data['waterfront'].values > 0] = 1
-------
data.view.value_counts()
0.0    19422
1.0     2112
Name: view, dtype: int64
1
data.sqft_basement.value_counts()
0.0    13280
1.0     8317
Name: sqft_basement, dtype: int64
1
data.waterfront.value_counts()
0.0    19075
1.0      146
Name: waterfront, dtype: int64
```
#### How were NaN, null, or immiscible values being handled?
Consequences of utilizing the dropna method wasn't feasible as it would have lead to a drop of more than 25% of the data set:
```python
# dropping NaN's axis= 1 col would remove to columns
# dropping NaN's axis= 0 rows would lose more than half of the dataframe! 
drop_nans = data.dropna(axis= 0)
display(data.shape) # original shape
drop_nans.shape # dropna utilized shape
(21597, 21)
(15762, 21)
```
sqft_basement had values of '?' which is being assumed as unknown so those values were changed to a value of 0 
```python
data['sqft_basement'] = data['sqft_basement'].replace('?', '0')
data['sqft_basement'] = data['sqft_basement'].astype(float)
```
#### Which, if any, independant variables were considered highly correlated and how was it handled?
The columns sqft_living and sqft_above are duplicated values, sqft_above means the living area above grade, not excluding any basement. The sqft_above was not being considered a feature of the model. 

<img src="images/data_scrub heatmap.png" width= 500>


#### How were possible outliers delt with?
A prevalent real estate metric used and missing in this data set is price_per_sqft. This was calculated and added to the data and utilized to find any underlying outliers like comparing dwelling prices by size and price. This can be searched by zipcode and comparing bedroom combinations (ie 2bdr vs 4bdr). Although it's completely feasible for tauted luxury dwelling to be marketed and priced accordingly. 

```python
# comparing price between home size 
# some outliers indicate that some 2br dwellings are more expensive than 5br's in the same zipcode and sqft. 

def scatter(df, zipcode):
    bed_a = data_scrub[(data.zipcode == zipcode) & (data.bedrooms == 2)] 
    bed_b = data_scrub[(data.zipcode == zipcode) & (data.bedrooms == 4)]
    
    plt.rcParams['figure.figsize'] = (12,7)
    plt.scatter(bed_a.sqft_living, bed_a.price, color = 'b', label = '2br', s= 50)
    plt.scatter(bed_b.sqft_living, bed_b.price, marker= '^', color = 'r', label = '4br', s= 50)

    plt.xlabel('Sqft_living')
    plt.ylabel('Price')
    plt.title(zipcode)
    plt.legend()

scatter(data, 98117)
```
<img src="images/98117.png" width=600>


```python
# use this line to run the nbconvert to readme.md github file and replace README.md
jupyter nbconvert --Markdown  readme.ipynb
```


      File "<ipython-input-9-740dfcf4d4cb>", line 2
        jupyter nbconvert --Markdown  readme.ipynb
                        ^
    SyntaxError: invalid syntax
    



```python

```
