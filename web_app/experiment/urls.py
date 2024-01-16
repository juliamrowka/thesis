from django.urls import path
from . import views
from . import forms

# app_name = "crm" #application namespace
urlpatterns = [
    path('', views.experiment, name='experiment'),
    path('upload/', views.model_form_upload, name='upload'),
    path('file/<int:pk>', views.choose_file, name='file'),
    path('delete-file/<int:pk>', views.delete_file, name='delete-file'),
    path('transformer/', views.transformer, name='transformer'),
    path('transformer/std', views.std, name='std'),
    path('transformer/minmax', views.minmax, name='minmax'),
    path('transformer/norm', views.norm, name='norm'),
    # path('transformer/pca', views.pca, name='pca'),
    path('estimator/', views.estimator, name='estimator'),
    path('estimator/ordinary-least-squares', views.ord_least_squares, name='ordinary-least-squares'),
    path('estimator/svm-regression', views.svm_regression, name='svm-regression'),
    path('estimator/nearest-neighbors-regression', views.nn_regression, name='nearest-neighbors-regression'),
    path('estimator/decision-tree-regression', views.dt_regression, name='decision-tree-regression'),
    path('estimator/categorical-naive-bayes', views.categorical_nb, name='categorical-naive-bayes'),
    path('estimator/svm-classification', views.svm_classification, name='svm-classification'),
    path('estimator/nearest-neighbors-classification', views.nn_classification, name='nearest-neighbors-classification'),
    path('estimator/decision-tree-classification', views.dt_classification, name='decision-tree-classification'),
    path('evaluation/', views.evaluation, name='evaluation'),
    path('evaluation/random-split', views.random_split, name='random-split'),
    path('evaluation/cross_validation', views.cross_validation, name='cross-validation'),
    path('compute', views.compute, name='compute'),
    path('models', views.show_models, name='models'),
    path('download/<int:pk>', views.download_model, name='download'),
    path('delete-model/<int:pk>', views.delete_model, name='delete-model'),
    path('transformer-up/', views.transformer_up, name='transformer-up'),
    path('transformer-down/', views.transformer_down, name='transformer-down'),
    path('delete-step/', views.delete_step, name='delete-step'),
    path('delete-est/', views.delete_est, name='delete-est'),
    path('delete-ev/', views.delete_ev, name='delete-ev'),

    # path('compute/save-model', views.save_model, name='save_model'),
    # path('documents', views.choose_file, name='choose_file')

]