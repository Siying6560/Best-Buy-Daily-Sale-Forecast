import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

warnings.filterwarnings("ignore")


a = pd.read_csv('data/data_cleaned.csv')
v = a.loc[a['SALES_DATE'] >= '2022-08-01']
valid_sku_list = list(v['Encoded_SKU_ID'].unique())




def get_sku_list(df, category_column):
    result = {}
    for cat in df[category_column].unique():
        s_list = list(df.loc[df[category_column] == cat, 'Encoded_SKU_ID'].unique())
        result[cat] = s_list
    return result


def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


# One-hot encoding for categorical variables
for col in ['Inventory', 'holiday', 'dayofweek']:
    dummy = pd.get_dummies(a[col], prefix=col, prefix_sep='_')
    a.drop(col, axis=1, inplace=True)
    a = a.join(dummy)


# Get sku group
sku_category = get_sku_list(a, 'ML_NAME')
print(sku_category.keys())
conn_car_sku = sku_category['ML - Connected Car']



# Drop some unwanted columns
col_to_drop = [
    'SUBCLASS_NAME', 'CLASS_NAME', 'ML_NAME', 'CATEGORY_NAME', 'RETAIL_PRICE', 'PROMO_PRICE', 'sales_diff', 'sell_price', 'promo',
    
    'year', 'month', 'day', 'weekofyear', 'dayofyear', 'quarter',
    
    'sales_lag1','sales_lag2','sales_ma14', 'cpi','cpi_lag1','cpi_lag2',
    
    'gscpi', 'gscpi_lag1', 'gscpi_lag2', 'UMCSENT', 'umcsent_lag1', 'umcsent_lag2',
    
    'Inventory_Constrained', 'Inventory_Fully-Stocked', 'Inventory_Moderate', 'Inventory_Out-of-Stock',
    
    'nonfarm','nonfarm_lag1','nonfarm_lag2','unemp','unemp_lag1','unemp_lag2',
    
    'holiday_Additional Day', 'holiday_Boxing Day', 'holiday_Canada Day', 'holiday_Christmas Additional Day',
    'holiday_Civic Holiday', 'holiday_Family Day', 'holiday_Good Friday', 'holiday_Labour Day', "holiday_New Year's Day",
    'holiday_Thanksgiving', 'holiday_Victoria Day', 'holiday_Christmas Day', 'holiday_Christmas Eve',

    'dayofweek_1', 'dayofweek_2', 'dayofweek_3', 'dayofweek_4', 'dayofweek_5', 'dayofweek_6', 'dayofweek_7',

    'holiday_Black Friday', 'holiday_Black Friday Pre-sale',
    'holiday_Cyber Monday'
    ]


a = a.drop(col_to_drop, axis=1)
print(a.columns)


# Change column data type
a['SALES_DATE'] = pd.to_datetime(a['SALES_DATE'])


def scale_minmax(df, exclude):
    # Min-Max Normalization
    x = df.loc[:, ~df.columns.isin(exclude)]
    x_scaled = (x - x.min()) / (x.max() - x.min())
    df = df.loc[:, exclude].join(pd.DataFrame(x_scaled))
    return df


def xgboost_per_sku(df, sku, random_position=0, cutoff_date=None):
    y = ['DAILY_UNITS']

    n = 7  # predict period

    # Rank by date descending for each sku
    df = df.loc[df['Encoded_SKU_ID'].isin([sku])]
    df['rank_sku'] = df.groupby('Encoded_SKU_ID')['SALES_DATE'].rank(ascending=False)
    
    # Filter features for sku category
    # if sku in conn_car_sku:
        # df = df.drop(['TOTALSA', 'totalsa_lag1', 'totalsa_lag2'], axis=1)


    if cutoff_date:
        df = df.loc[df['SALES_DATE'] >= cutoff_date]

    # Scale data
    # df = scale_minmax(df, exclude=['Encoded_SKU_ID', 'SALES_DATE', 'DAILY_UNITS', 'rank_sku', 'sales_diff'])
    # df.to_csv('test.csv')

    # Test, train manual split
    k = random_position
    test = df.loc[(df['rank_sku'] <= n+k) & (df['rank_sku'] > k)]
    train = df.loc[df['rank_sku'] > n+k]



    # ARIMA
    # arima = auto_arima(train[y[0]], start_p=1, start_q=1, test='adf', max_p=2, max_q=2, seasonal=True,
                       # error_action='ignore', suppress_warnings=True, stepwise=True)
    # train['arima_resid'] = arima.resid().array

    # Expo Smoothing
    expo = ExponentialSmoothing(train[y[0]], seasonal='add', seasonal_periods=7).fit(smoothing_level=0.85, remove_bias=True)
    train['expo'] = expo.fittedvalues
    train['expo_resid'] = train['expo'] - train['DAILY_UNITS']
    train = train.drop('expo', axis=1)
    
    expo_pred = expo.forecast(n)
    expo_pred[expo_pred<0] = 0
    

    # Train Xy split
    columns_to_drop_model = ['DAILY_UNITS', 'sales_diff', 'SALES_DATE', 'rank_sku', 'expo_resid']
    last_record = train.loc[train['rank_sku'] == n+1, 'DAILY_UNITS'].values


    X_train = train.loc[:, ~train.columns.isin(y + columns_to_drop_model)]
    y_train = train['expo_resid']
    # Test split (X_test is created in recursive prediction process)
    y_test = test['DAILY_UNITS']


    # Train XGBoost model
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.2, gamma=0)  # XGBoost
    xgb.fit(X_train, y_train, verbose=False)


    # Combine Expo smoothing and XGB
    X_test = test.loc[:, ~test.columns.isin(y + columns_to_drop_model)]
    
    y_pred_resid = xgb.predict(X_test)

    y_pred = expo_pred - y_pred_resid
    y_pred = np.round(y_pred)


    rmse = np.sqrt(mean_squared_error(y_pred, y_test))


    # print("SKU: " + str(sku) + ", Rooted Mean Squared Error : " + str(rmse))

    return y_pred, y_test, rmse


def cross_test(random_position, cutoff_date):
    pred = []
    test = []
    sku_result = dict()
    for sku in valid_sku_list:
    # for sku in [469, 347, 557, 237]:
        pred_i, test_i, mse = xgboost_per_sku(a, sku, random_position=random_position, cutoff_date='2021-03-01')
        pred.append(pred_i)
        test.append(test_i)
        sku_result[sku] = mse

    # Save result
    result_df = pd.DataFrame(sku_result.items(), columns=['sku', 'rmse'])
    file_name = "expo_resid_xgb_"+str(cutoff_date)+".csv"
    result_df.to_csv('result/'+file_name, index=False)

    rmse = np.sqrt(mean_squared_error(pred, test))

    print("Method: EXPO + RESID, Overall Rooted Mean Squared Error : " + str(rmse))
    return rmse, sku_result


for r in [0]:
    cross_test(r, '2022-05-01')



