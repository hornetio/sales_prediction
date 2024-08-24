from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import os
import pandas as pd
import pickle
import numpy as np


def index(request):
    return render(request, 'predictor/index.html')


def predict_sales(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']

        # Save the uploaded file to the server
        file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Process the uploaded file
        try:
            df = pd.read_csv(file_path,encoding='iso-8859-1', na_values=[''], keep_default_na=False)
            df_prepared = prepareData(df)
            print("Prepare data done")

            # Load the model and make predictions
            model_path = os.path.join(settings.MEDIA_ROOT, 'models', 'model.pkl')
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
                print("Start predict")
                predictions = model.predict(df_prepared)
                print("End predict")
                df['Sales'] = predictions

                # Save predictions to a new CSV file
                result_file_path = os.path.join(settings.MEDIA_ROOT, 'predictions.csv')
                df.to_csv(result_file_path, index=False)

            # Send the predictions file to the user
            with open(result_file_path, 'rb') as f:
                response = HttpResponse(f.read(), content_type='text/csv')
                response['Content-Disposition'] = 'attachment; filename="predictions.csv"'
                return response

        except Exception as e:
            return HttpResponse(f"Error processing file: {e}", status=500)

    return HttpResponse('Invalid request method or missing file', status=400)


def prepareData(df):
    df_form = df.copy()

    # Загрузка и объединение дополнительных данных о магазинах
    store_data = pd.read_csv(
        os.path.join(settings.MEDIA_ROOT, 'store', 'store.csv'),
        encoding='iso-8859-1', na_values=[''], keep_default_na=False
    )
    store_data = store_data.fillna("0")
    df_form = pd.merge(df_form, store_data, on='Store')

    # Обработка пропущенных значений
    df_form.dropna()

    # Преобразование 'Date' в формат datetime и извлечение года, месяца и дня
    if 'Date' in df_form.columns:
        df_form['Date'] = pd.to_datetime(df_form['Date'], errors='coerce')
        df_form['Year'] = df_form['Date'].dt.year - 2000
        df_form['Month'] = df_form['Date'].dt.month
        df_form['Day'] = df_form['Date'].dt.day
        del df_form['Date']
    else:
        # Предоставление значений по умолчанию для отсутствующего столбца 'Date'
        df_form['Year'] = 0
        df_form['Month'] = 0
        df_form['Day'] = 0




    # Определение числовых столбцов как всех оставшихся столбцов

    numeric_columns = ['Open', 'Promo', 'SchoolHoliday',
                  'CompetitionDistance', 'CompetitionOpenSinceMonth',
                  'Promo2', 'Promo2SinceWeek','Promo2SinceYear','Year', 'Month','Day','CompetitionOpenSinceYear',
                  ]

    for column in numeric_columns:
        if column in df_form.columns:
            df_form[column] = pd.to_numeric(df_form[column], errors='coerce')

    #df_form['CompetitionOpenSinceYear'] -= 2000
    #df_form.loc[df_form['Promo2SinceYear'] > 2000, 'Promo2SinceYear'] -= 2000

    categorical_columns = ['Store', 'DayOfWeek', 'StateHoliday',
       'StoreType', 'Assortment',
         'PromoInterval']


    for column in categorical_columns:
        if column in df_form.columns:
            df_form[column] = df_form[column].to_string()



    # Дополнительное преобразование всех категориальных признаков в строки
    for column in categorical_columns:
        if column in df_form.columns:
            df_form[column] = df_form[column].astype(str)

    print(df_form.columns)
    return df_form



