# predictor/views.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle

from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings

# Определение категориальных и числовых признаков
categorical_column = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']
num_column = ['Sales', 'Customers', 'Open', 'Promo', 'SchoolHoliday',
                      'CompetitionDistance', 'CompetitionOpenSinceMonth',
                      'Promo2', 'Promo2SinceWeek'
                      ]

date_column = ['Promo2SinceYear', 'CompetitionOpenSinceYear', "Date"]


def index(request):
    return render(request, 'predictor/index.html')


def predict_sales(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']

        # Используйте путь к папке data в корне проекта
        file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)

        # Убедитесь, что папка существует
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

        # Сохранение файла в папку data
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Обработка загруженного файла
        try:
            df = pd.read_csv(file_path)
            df = prepareData(df)
            # Загрузите модель из файла

            with open(settings.MEDIA_ROOT+'\\models\\model.pkl', 'rb') as file:
                model = pickle.load(file)

                results = model.predict(df)
                result_file_path = os.path.join(settings.MEDIA_ROOT, 'predictions.csv')
                results.to_csv(result_file_path, index=False)

            # Отправка файла с результатами пользователю
            with open(result_file_path, 'rb') as f:
                response = HttpResponse(f.read(), content_type='text/csv')
                response['Content-Disposition'] = 'attachment; filename="predictions.csv"'
                return response
        except Exception as e:
            return HttpResponse(f"Error processing file: {e}", status=500)

    return HttpResponse('Invalid request method or missing file', status=400)

def prepareData(df):
    df_form = df.copy()

    store_data = pd.read_csv(
        settings.MEDIA_ROOT+'\\store\\store.csv',
        encoding='iso-8859-1', na_values=[''], keep_default_na=False)
    store_data = store_data.fillna("0")
    df = pd.merge(df_form, store_data, on='Store')

    df_form.replace({' - ': np.nan, '\\N': np.nan, 'NaN': np.nan}, inplace=True)
    df_form = df_form.dropna()

    # Факторизация категориальных столбцов
    encoder = OneHotEncoder(sparse_output=False)
    encoded_df_list = []

    for name in categorical_column:
        # Применение OneHotEncoder к целевому столбцу
        encoded_columns = encoder.fit_transform(df_form[[name]])
        # Преобразование закодированных данных в DataFrame
        encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out([name]))
        encoded_df_list.append(encoded_df)

    # Объединение закодированных столбцов с исходным DataFrame (удалив при этом исходные столбцы)
    df_form = df_form.drop(columns=categorical_column)
    df_form = pd.concat([df_form] + encoded_df_list, axis=1)

    df_form['Year'] = df_form['Date'].dt.year - 2000
    df_form['Month'] = df_form['Date'].dt.month
    df_form['Day'] = df_form['Date'].dt.day
    del df_form['Date']
    # Приведение всех числовых столбцов к типу int
    for column in df_form.columns:
        df_form[column] = pd.to_numeric(df_form[column], errors='coerce')

    df_form = df_form.dropna()

    df_form['CompetitionOpenSinceYear'] -= 2000
    df_form.loc[df_form['Promo2SinceYear'] > 2000, 'Promo2SinceYear'] -= 2000

    return df_form

