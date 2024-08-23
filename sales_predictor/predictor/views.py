# predictor/views.py
import os
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings


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
            results = process_data(df)
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


def process_data(df):
    # Пример обработки данных и создания результатов
    df['Predictions'] = df['Sales'] * 1.1  # Простое увеличение значений для примера
    return df
