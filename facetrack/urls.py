from django.urls import path
from . import views

urlpatterns = [
    path("fetch-base-images/", views.fetch_base_images, name="fetch_base_images"),
]
