import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
from prophet import Prophet

warnings.filterwarnings("ignore")

a = pd.read_csv('data/data_cleaned.csv')
t = a.loc[a['SALES_DATE'] < '2022-08-01']
v = a.loc[a['SALES_DATE'] >= '2022-08-01']

holidays = pd.read_csv('external/holiday_sales.csv')
holidays.columns = ['ds', 'holiday']


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


def prophet_predict(train, test, cutoff_date):
    y_test = []
    y_pred = []

    train.loc[:, 'SALES_DATE'] = pd.to_datetime(train['SALES_DATE'])
    train = train.loc[train['SALES_DATE'] >= cutoff_date]
    test.loc[:, 'SALES_DATE'] = pd.to_datetime(train['SALES_DATE'])

    sku_result = dict()
    # for sku in [1, 2]:
    for sku in list(test['Encoded_SKU_ID'].unique()):

        sales_train = train.loc[train['Encoded_SKU_ID'] == sku, ['SALES_DATE', 'DAILY_UNITS']]
        sales_train.columns = ['ds', 'y']
        sales_train.loc[sales_train['y'] < 0, 'y'] = 0

        sales_test = test.loc[test['Encoded_SKU_ID'] == sku, ['SALES_DATE', 'DAILY_UNITS']]
        sales_test.columns = ['ds', 'y']

        # Prophet
        model = Prophet(seasonality_mode='additive', holidays=holidays)
        model.fit(sales_train)
        future = model.make_future_dataframe(7)
        pred = model.predict(future)['yhat'].tail(7).values
        pred[pred < 0] = 0
        pred = np.round(pred)

        rmse = np.sqrt(mean_squared_error(sales_test['y'], pred))

        print("SKU: "+str(sku)+", RMSE: "+str(rmse))

        # Save result
        sku_result[sku] = rmse
        result_df = pd.DataFrame(sku_result.items(), columns=['sku', 'rmse'])
        file_name = "prophet_"+str(cutoff_date)+".csv"
        result_df.to_csv('result/' + file_name, index=False)

        y_pred.append(list(pred))
        y_test.append(list(sales_test['y'].values))


    print("Overall RMSE: "+str(np.sqrt(mean_squared_error(y_pred, y_test))))


prophet_predict(t, v, '2022-06-01')