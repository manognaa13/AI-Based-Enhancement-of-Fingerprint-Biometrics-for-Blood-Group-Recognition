from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('register', views.register, name='register'),
    path('login', views.login, name='login'),
    path('home', views.home, name='home'),
    path('home2', views.home2,name='home2'),
   
]


