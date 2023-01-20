import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings


warnings.filterwarnings("ignore")


a = pd.read_csv('data/data_cleaned.csv')
v = a.loc[a['SALES_DATE'] >= '2022-08-01']
valid_sku_list = list(v['Encoded_SKU_ID'].unique())


def get_sku_list(df, category_column):
    result = []
    for cat in df[category_column].unique():
        s_list = list(df.loc[df[category_column] == cat, 'Encoded_SKU_ID'].unique())
        result.append(s_list)
    return result

sku_list = get_sku_list(a, 'SUBCLASS_NAME')


# One-hot encoding for categorical variables
def one_hot(df, columns):
    for col in columns:
        dummy = pd.get_dummies(df[col], prefix=col, prefix_sep='_')
        df.drop(col, axis=1, inplace=True)
        df = df.join(dummy)
    return df


a = one_hot(a, ['SUBCLASS_NAME', 'CLASS_NAME', 'ML_NAME', 'CATEGORY_NAME', 'Inventory', 'holiday', 'year', 'month',
            'dayofweek'])

# Drop some unwanted columns
col_to_drop = ['RETAIL_PRICE', 'PROMO_PRICE', 'day', 'weekofyear', 'dayofyear', 'quarter']

a = a.drop(col_to_drop, axis=1)

n = 7

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


def scale_minmax(df, exclude):
    # Min-Max Normalization
    x = df.loc[:, ~df.columns.isin(exclude)]
    x_scaled = (x - x.min()) / (x.max() - x.min())
    df = df.loc[:, exclude].join(pd.DataFrame(x_scaled))
    return df


def xgboost_per_sku(df, sku, random_position=0, y_to_train='sales'):
    if y_to_train == 'sales':
        y = ['DAILY_UNITS']
    elif y_to_train == 'diff':
        y = ['sales_diff']

    # Rank by date descending for each sku
    df['rank_sku'] = df.sort_values(['Encoded_SKU_ID', 'SALES_DATE']).groupby('Encoded_SKU_ID')['SALES_DATE'].rank(ascending=False)
    if type(sku) == list:
        df = df.loc[df['Encoded_SKU_ID'].isin(sku)]
    else:
        df = df.loc[df['Encoded_SKU_ID'].isin([sku])]

    # Scale data
    # df = scale_minmax(df, exclude=['Encoded_SKU_ID', 'SALES_DATE', 'DAILY_UNITS', 'rank_sku', 'sales_diff'])
    # df.to_csv('test.csv')

    # Test, train manual split
    k = random_position
    test = df.loc[(df['rank_sku'] <= n+k) & (df['rank_sku'] > k)]
    pre_test = df.loc[(df['rank_sku'] <= n+k+28) & (df['rank_sku'] > k)].sort_values(['Encoded_SKU_ID', 'rank_sku'])
    pre_test = pre_test.reset_index(drop=True)
    train = df.loc[(df['rank_sku'] > n+k) & (df['rank_sku'] < 540+n+k)]

    # SKU ID one-hot encoding
    test = one_hot(test, ['Encoded_SKU_ID'])
    pre_test = one_hot(pre_test, ['Encoded_SKU_ID'])
    train = one_hot(train, ['Encoded_SKU_ID'])


    # Train split
    columns_to_drop_model = ['DAILY_UNITS', 'sales_diff', 'SALES_DATE', 'rank_sku']
    X_train = train.loc[:, ~train.columns.isin(y + columns_to_drop_model)]
    y_train = train[y[0]]
    # Test split (X_test is created in recursive prediction process)
    y_test = test['DAILY_UNITS']

    # Train the model
    model = XGBRegressor()  # XGBoost
    model.fit(X_train, y_train, verbose=False)

    # Recursive prediction
    # Add lag
    lag_column_name = []
    for j in np.arange(1, 3, 1):
        lag_name = 'sales_lag' + str(j)
        lag_column_name.append(lag_name)

    dependent_columns = lag_column_name + ['sales_ma14', 'competitor_diff']
    sales_columns = ['DAILY_UNITS', 'sales_diff']
    pre_test.loc[pre_test['rank_sku'] <= n+k, dependent_columns + sales_columns] = np.nan
    test_ind = pre_test.loc[pre_test['rank_sku'] <= n+k].index

    for i in np.arange(1, n+1, 1)[::-1]:
        ind = pre_test.loc[pre_test['rank_sku'] == i+k].index

        # Impute dependent values
        # pre_test.loc[ind, 'sales_diff'] = pre_test.loc[ind, 'DAILY_UNITS'].array - pre_test.loc[ind + 1, 'DAILY_UNITS'].array
        # pre_test.loc[ind, 'sales_ma7'] = pre_test.loc[:, 'DAILY_UNITS'][::-1].rolling(7, closed='left').mean().loc[ind].array
        pre_test.loc[ind, 'sales_ma14'] = pre_test.loc[:, 'DAILY_UNITS'][::-1].rolling(14, closed='left').mean().loc[ind].array
        # pre_test.loc[ind, 'sales_ma28'] = pre_test.loc[:, 'DAILY_UNITS'][::-1].rolling(28, closed='left').mean().loc[ind].array
        for j in np.arange(1, 3, 1):
            pre_test.loc[ind, 'sales_lag'+str(j)] = pre_test.loc[ind + j, 'DAILY_UNITS'].array

        # Predict
        test_i = pre_test.loc[ind]
        X_test_i = test_i.loc[:, ~test_i.columns.isin(y + columns_to_drop_model)]
        pred_i = model.predict(X_test_i)

        if y_to_train == 'sales':
            pre_test.loc[ind, 'DAILY_UNITS'] = pred_i
        elif y_to_train == 'diff':
            pre_test.loc[ind, 'DAILY_UNITS'] = pred_i + pre_test.loc[ind + 1, 'DAILY_UNITS'].array

    y_pred = pre_test.loc[test_ind, 'DAILY_UNITS']
    y_pred[y_pred < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_pred, y_test))


    print("SKU: " + str(sku) + ", Rooted Mean Squared Error : " + str(rmse))
    # print(y_pred, "\n", y_test)

    return y_pred, y_test, rmse


def cross_test(method, random_position):
    pred = []
    test = []
    sku_result = dict()
    for sku in valid_sku_list:
    # for sku in sku_list:
    # for sku in [[469], [237, 557], [1]]:
        pred_i, test_i, rmse = xgboost_per_sku(a, sku, random_position=random_position, y_to_train=method)
        pred.append(list(pred_i.array))
        test.append(list(test_i.array))
        sku_result[str(sku)] = rmse

    # Save result
    result_df = pd.DataFrame(sku_result.items(), columns=['sku', 'rmse'])
    file_name = "xgb.csv"
    result_df.to_csv('result/'+file_name, index=False)

    pred = flatten_list(pred)
    test = flatten_list(test)

    print("Method: " + str(method) + ", Total Mean Squared Error : " + str(mean_squared_error(pred, test)))
    return mean_squared_error(pred, test), sku_result


for m in ['sales']:
    for r in [0]:
        cross_test(m, r)
