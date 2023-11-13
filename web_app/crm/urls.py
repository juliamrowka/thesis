from django.urls import path
from . import views

# app_name = "crm" #application namespace
urlpatterns = [
    path('', views.home, name='home'),
    # path('login/', views.login_user, name='login'),
    path('logout/', views.logout_user, name='logout'),

]