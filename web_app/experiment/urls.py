from django.urls import path
from . import views

# app_name = "crm" #application namespace
urlpatterns = [
    path('upload/', views.model_form_upload, name='upload'),
    path('', views.experiment, name='experiment'),

]