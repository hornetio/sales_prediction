import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
test_df = pd.read_csv('D:/Valerian/Documents/OneDrive/Python/ДопОбр Анализ данных/Практика Нетодология/sales_prediction/Команда_11/test.csv', encoding='iso-8859-1',na_values=[''], keep_default_na=False)

train_df = pd.read_csv('D:/Valerian/Documents/OneDrive/Python/ДопОбр Анализ данных/Практика Нетодология/sales_prediction/Команда_11/train.csv', parse_dates=['Date'], low_memory=False, na_values=[''], keep_default_na=False)

# Загрузка данных из файла store.csv
add_df = pd.read_csv('D:/Valerian/Documents/OneDrive/Python/ДопОбр Анализ данных/Практика Нетодология/sales_prediction/Команда_11/store.csv', encoding='iso-8859-1',na_values=[''], keep_default_na=False)
train_df.isna().sum()
test_df.isna().sum()
test_df.fillna(0,inplace=True)
add_df.isna().sum()
add_df.loc[add_df['Promo2'] == 0, ['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']] = 0
add_df['CompetitionDistance'].fillna(add_df['CompetitionDistance'].mean(), inplace=True)
add_df['CompetitionOpenSinceMonth'].fillna(int(add_df['CompetitionOpenSinceMonth'].mean()), inplace=True)
add_df['CompetitionOpenSinceYear'].fillna(int(add_df['CompetitionOpenSinceYear'].mean()), inplace=True)
merged_test = pd.merge(test_df, add_df, on='Store')
merged_train = pd.merge(train_df, add_df, on='Store')
merged_train['Date'] = pd.to_datetime(merged_train['Date'])
merged_train = merged_train.sort_values(by='Date')

merged_train['Year'] = merged_train['Date'].dt.year
merged_train['Month'] = merged_train['Date'].dt.month
merged_train['Day'] = merged_train['Date'].dt.day

merged_train=merged_train.drop(['Date'],axis=1)
customer_df=merged_train.drop(['Sales'],axis=1)
customer_df[[   'CompetitionOpenSinceMonth',
       'Promo2SinceWeek']] = customer_df[['CompetitionOpenSinceMonth',
       'Promo2SinceWeek']].astype(int)
cat_features=['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday',
       'SchoolHoliday', 'StoreType', 'Assortment','CompetitionOpenSinceMonth', 'Promo2',
       'Promo2SinceWeek', 'PromoInterval', 'Month',]

X=customer_df.drop(['Customers'],axis=1)
y=customer_df['Customers']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostRegressor(iterations=100,
                          learning_rate=0.1,
                          depth=10,
                          cat_features=cat_features,
                          random_seed=42,
                          verbose=100,
                          has_time=True)

model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

import numpy as np
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')
merged_test[[   'CompetitionOpenSinceMonth',
       'Promo2SinceWeek','Open']] = merged_test[['CompetitionOpenSinceMonth',
       'Promo2SinceWeek','Open']].astype(int)
merged_test=merged_test.drop('Id',axis=1)
merged_test['Date'] = pd.to_datetime(merged_test['Date'])
merged_test = merged_test.sort_values(by='Date')
merged_test['Year'] = merged_test['Date'].dt.year
merged_test['Month'] = merged_test['Date'].dt.month
merged_test['Day'] = merged_test['Date'].dt.day
merged_test=merged_test.drop(['Date'],axis=1)
merged_test['Customers']=model.predict(merged_test)
sales_df[[   'CompetitionOpenSinceMonth',
       'Promo2SinceWeek']] = sales_df[['CompetitionOpenSinceMonth',
       'Promo2SinceWeek']].astype(int)
X=sales_df.drop(['Sales'],axis=1)
y=sales_df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = CatBoostRegressor(iterations=100,
                          learning_rate=0.1,
                          depth=10,
                          cat_features=cat_features,
                          random_seed=42,
                          verbose=100,
                          has_time=True)

model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

import numpy as np
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')
final=merged_test[['Store', 'DayOfWeek', 'Customers', 'Open', 'Promo',
       'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
       'CompetitionDistance', 'CompetitionOpenSinceMonth',
       'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
       'Promo2SinceYear', 'PromoInterval', 'Year', 'Month', 'Day']]
final['Sales']=model.predict(final)
import pickle

# Сохраните модель в файл
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)












