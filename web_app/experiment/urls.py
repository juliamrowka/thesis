from django.urls import path
from . import views
from . import forms

# app_name = "crm" #application namespace
urlpatterns = [
    path('', views.experiment, name='experiment'),
    path('upload/', views.model_form_upload, name='upload'),
    path('file/<int:pk>', views.choosen_file, name='file'),
    path('transformer/', views.transformer, name='transformer'),
    path('transformer/std', views.std, name='std'),
    path('transformer/minmax', views.minmax, name='minmax'),
    path('transformer/norm', views.norm, name='norm'),
    path('transformer/pca', views.pca, name='pca'),
    # path('documents', views.choose_file, name='choose_file')

]