# Best Buy Daily Sale Forecast

**Background**: Accurate sales forecasting is crucial for Best Buy in many ways (e.g., inventory planning and pricing). This needs to be extended to the items not sold in large quantities. The challenge is to detect real patterns due to presence of many zero sales days. Efficient forecasting models are desired as there are a huge number of products in this category.

**Objective**: Formulate a weeklong forecast the slow-selling SKUs with accuracy and execution time as the most important considerations. Expected output is daily predictions for the seven days of the week just after the end of the training data set.

**Approach**: After trying different models including both time series model such as Exponential smoothing, Facebook Prophet and ensemble models such as random forest tree and XGboost, we decided to train exponential smoothing model to predict the baseline and train XGboost model to predict the residual.

**Result**: Report a RMSE 3.06 on validtion dataset.
