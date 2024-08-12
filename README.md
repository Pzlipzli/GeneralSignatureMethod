# GeneralSignatureMethod
A general method of processing data and using signature to extract time characteristics from the data.
# Attention:
1. Only DataLoader is runnable.
2. Before running, change the data path in Settings.
3. The data should be csv files with headers of '开盘时间', '开盘价', '最高价', '最低价', '收盘价', '成交量', '货币对' in simplified Chinese.
   You can change the acceptable headers in the DataLoader file.
4. The generated file will be saved in ./config/data. This program generates one csv file for one day. The headers are alpha_0, alpha_1,
   alpha_2, ..., alpha_n, token, openTime, closeTime.
5. The original time is UTF+8.
