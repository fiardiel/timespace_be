# authentication/urls.py
from django.urls import path
from .views import imagerecord_list

urlpatterns = [
    path('imagerecords/', imagerecord_list, name='imagerecord_list'),
]
