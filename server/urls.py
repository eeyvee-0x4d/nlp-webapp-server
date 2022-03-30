"""server URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include,path

from backend import views

urlpatterns = [
    path('upload/', views.upload, name='upload'),
    path('data_overview/', views.data_overview, name='data_overview'),
    path('sentiment_overview/', views.sentiment_overview, name='sentiment_overview'),
    path('sentiment_trend/', views.sentiment_trend, name='sentiment_trend'),
    path('all_data/', views.all_data, name='all_data'),
    path('model_stats/', views.model_stats, name='model_stats'),
]
