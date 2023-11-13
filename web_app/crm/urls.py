from django.urls import path
from . import views

# app_name = "crm" #application namespace
urlpatterns = [
    path('', views.home, name='home'),

]