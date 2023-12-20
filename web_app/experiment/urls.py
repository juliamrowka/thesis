from django.urls import path
from . import views
from . import forms

# app_name = "crm" #application namespace
urlpatterns = [
    path('', views.experiment, name='experiment'),
    path('upload/', views.model_form_upload, name='upload'),
    path('file/<int:pk>', views.choose_file, name='file'),
    path('delete/<int:pk>', views.delete_file, name='delete'),
    path('transformer/', views.transformer, name='transformer'),
    path('transformer/std', views.std, name='std'),
    path('transformer/minmax', views.minmax, name='minmax'),
    path('transformer/norm', views.norm, name='norm'),
    path('transformer/pca', views.pca, name='pca'),
    path('estimator', views.estimator, name='estimator'),
    path('estimator/ordinary-least-squares', views.ord_least_squares, name='ordinary-least-squares'),
    # path('estimator', views.estimator, name='estimator'),
    # path('estimator', views.estimator, name='estimator'),
    # path('estimator', views.estimator, name='estimator'),
    # path('estimator', views.estimator, name='estimator'),
    # path('estimator', views.estimator, name='estimator'),
    # path('estimator', views.estimator, name='estimator'),
    # path('estimator', views.estimator, name='estimator'),
    # path('estimator', views.estimator, name='estimator'),
    # path('documents', views.choose_file, name='choose_file')

]