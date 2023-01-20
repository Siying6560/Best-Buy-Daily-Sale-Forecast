import numpy as np
import pandas as pd
import warnings
import os
from datetime import datetime
warnings.filterwarnings("ignore")


c = pd.read_csv('data/Hackathon Data.csv')
b = pd.read_csv('data/Validation_Data.csv')
a = pd.concat([c, b], ignore_index=True)


# Assign data type, sort by SKU and date
for col in ['RETAIL_PRICE', 'PROMO_PRICE', 'COMPETITOR_PRICE', 'DAILY_UNITS']:
    a[col] = a[col].astype(str)
    a[col] = a[col].str.replace(",", "")
    a[col] = pd.to_numeric(a[col], errors='coerce')
a['SALES_DATE'] = pd.to_datetime(a['SALES_DATE'])
a.sort_values(['Encoded_SKU_ID', 'SALES_DATE'], inplace=True)


# Impute missing date
saledate_beg = a.groupby('Encoded_SKU_ID')['SALES_DATE'].min().reset_index()
saledate_beg.rename(columns = {'SALES_DATE':'SALES_DATE_BEGIN'}, inplace = True)
# date_series = pd.date_range('2017-05-28', periods=1891, freq ='D')

date_series = pd.date_range('2017-05-28', periods=1898, freq ='D')

date_table = date_series.to_frame()
date_table.columns = ['SALES_DATE']
date_table.reset_index(drop = True, inplace = True)
sku = pd.DataFrame(a.Encoded_SKU_ID.unique(), columns=['Encoded_SKU_ID'])  # create sku table
sku.sort_values('Encoded_SKU_ID', inplace=True)
temp = sku.merge(date_table, how ='cross')  # cross merge date_table with sku table
temp2 = temp.merge(saledate_beg, how='left', on ='Encoded_SKU_ID')

def drop_notexist_date(x):
    # drop the row when sale date is prior to the sale begin date for each sku
    if x.SALES_DATE < x.SALES_DATE_BEGIN:
        x.SALES_DATE = np.nan
    return x.SALES_DATE

temp2['SALES_DATE'] = temp2.apply(drop_notexist_date, axis=1)
temp3 = temp2[temp2.SALES_DATE.notnull()]
temp3.drop('SALES_DATE_BEGIN', axis=1, inplace=True)  # drop 'SALES_DATE_BEGIN' column and temp3 only has 'Encoded_SKU_ID' and 'SALES_DATE' column
a = pd.merge(temp3, a, on=['Encoded_SKU_ID', 'SALES_DATE'], how='left')  # merge temp3 with df

a.loc[a['DAILY_UNITS'].isna(), 'DAILY_UNITS'] = 0  # Set daily sales to zero for imputed rows


# Add discount info, generate real sell price and promotion
a['promo'] = a.RETAIL_PRICE - a.PROMO_PRICE
a['promo_ratio'] = a['promo'] / a['RETAIL_PRICE']
a.loc[a['promo'].isna(), 'promo'] = 0
a['sell_price'] = a.RETAIL_PRICE
a['sell_price'][a.PROMO_PRICE.notnull()] = a.PROMO_PRICE


# Add difference between competitor
a['competitor_diff'] = a['COMPETITOR_PRICE'] - a['sell_price']


# Add date features: year, month, day, dayofweek, quarter, weekofyear, dyaofyear
a['year'] = a['SALES_DATE'].dt.year
a['month'] = a['SALES_DATE'].dt.month
a['day'] = a['SALES_DATE'].dt.day
a['quarter']= a['SALES_DATE'].dt.quarter
a['dayofweek'] = a['SALES_DATE'].dt.dayofweek+1
a['weekofyear'] = a['SALES_DATE'].dt.weekofyear
a['dayofyear'] = a['SALES_DATE'].dt.dayofyear



# Add lag 1 and lag 2 features
for i in np.arange(1, 3, 1):
    a['sales_lag'+str(i)] = a.groupby('Encoded_SKU_ID')['DAILY_UNITS'].shift(i)


# Add moving average of 1 week, 14 days, 28 days
a["rank_sku"] = a.sort_values(['Encoded_SKU_ID', 'SALES_DATE']).groupby('Encoded_SKU_ID')['SALES_DATE'].rank(method="first", ascending=True)

def moving_average(colname, days):
    a[colname] = a.sort_values(['Encoded_SKU_ID', 'SALES_DATE'])['DAILY_UNITS'].rolling(days, closed='left').mean()
    a.loc[a['rank_sku'] <= days-1, colname] = np.nan


for g in [('sales_ma14', 14)]:
    moving_average(g[0], g[1])

a = a.drop('rank_sku', axis=1)


# Calculate difference
a['sales_diff'] = a['DAILY_UNITS'] - a['sales_lag1']

# Drop nan rows
a.drop(a.loc[a['sales_ma14'].isna() | a['SUBCLASS_NAME'].isna() | a['sales_diff'].isna()].index, inplace=True)


# Add external data sources
file_list = ['cpi.csv', 'gscpi.csv', 'nonfarm.csv', 'totalsa.csv', 'umcsent.csv', 'unemp.csv']
for file in file_list:
    feature_name = file.split(".")[0]
    data = pd.read_csv('external/' + file)
    data[feature_name + '_lag1'] = data.iloc[:, 1].shift(1)
    data[feature_name + '_lag2'] = data.iloc[:, 1].shift(2)
    data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])
    data['year'] = data.iloc[:, 0].dt.year
    data['month'] = data.iloc[:, 0].dt.month
    a = pd.merge(a, data.iloc[:, 1:], how='left', on=['year', 'month'])

day_file_list = ['holiday.csv']
for file in day_file_list:
    feature_name = file.split(".")[0]
    data = pd.read_csv('external/' + file)
    data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])
    a = pd.merge(a, data, how='left', left_on='SALES_DATE', right_on='date')
    a.loc[a[feature_name].isna(), feature_name] = 'NA'
a.drop('date', axis=1, inplace=True)


# a.to_csv('data/data_cleaned_train.csv', index=False)
a.to_csv('data/data_cleaned.csv', index=False)