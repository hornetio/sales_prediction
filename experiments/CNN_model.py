import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Загрузка данных
df = pd.read_csv('D:/Valerian/Documents/OneDrive/Python/ДопОбр Анализ данных/Практика Нетодология/sales_prediction/Команда_11/train.csv', parse_dates=['Date'], low_memory=False)

# Удаление столбца 'Customers'
df = df.drop(columns=['Customers'])

# Преобразование даты в datetime
df['Date'] = pd.to_datetime(df['Date'])

# Обработка пропусков и ненужных значений
df = df.fillna(0)  # Или другой метод обработки пропусков

# Добавление дополнительных признаков (например, день недели, месяц)
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month


# Преобразование категориальных данных в числовые
df = pd.get_dummies(df, columns=['StateHoliday'])

# Сортировка данных по дате
df = df.sort_values(by='Date')

# Удаление ненужных столбцов
df = df[['Store', 'Sales' , 'Open', 'Promo', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c', 'SchoolHoliday', 'DayOfWeek', 'Month']]

# Нормализация признаков
scaler = StandardScaler()
df[[ 'Open', 'Promo', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c', 'SchoolHoliday', 'DayOfWeek', 'Month']] = scaler.fit_transform(df[[ 'Open', 'Promo', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c', 'SchoolHoliday', 'DayOfWeek', 'Month']])

print(df.head())
# Разделение данных на обучающую и тестовую выборки
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

seq_length = 30  # Длина окна

# Преобразование данных в массивы для CNN
train_sequences = create_sequences(train_df.values, seq_length)
test_sequences = create_sequences(test_df.values, seq_length)

X_train = train_sequences[:, :-1]
y_train = train_sequences[:, -1, 1]  # Sales is the second column
X_test = test_sequences[:, :-1]
y_test = test_sequences[:, -1, 1]

# Преобразование типов
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
X_test = X_test.astype('float32')
y_test = y_test.astype('float32')

# Создание модели
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))  # Предсказание объема продаж

model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
history = model.fit(X_train, y_train, epochs=7, validation_data=(X_test, y_test))

# Прогнозирование на тестовых данных
y_pred = model.predict(X_test)

# Вычисление метрик

mae_cnn = mean_absolute_error(y_test, y_pred)
mse_cnn = mean_squared_error(y_test, y_pred)
rmse_cnn = np.sqrt(mse_cnn)
r2_cnn = r2_score(y_test, y_pred)

# Вывод значений метрик
print(f"Среднее абсолютное отклонение (MAE) на тестовых данных (CNN): {mae_cnn}")
print(f"Корень средней квадратичной ошибки (RMSE) на тестовых данных (CNN): {rmse_cnn}")
print(f"Среднеквадратичная ошибка (MSE) на тестовых данных (CNN): {mse_cnn}")
print(f"Коэффициент детерминации (R2) на тестовых данных (CNN): {r2_cnn}")
