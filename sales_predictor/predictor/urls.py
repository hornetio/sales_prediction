# predictor/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Главная страница приложения
    path('predict/', views.predict_sales, name='predict_sales'),  # Страница для предсказаний
]
