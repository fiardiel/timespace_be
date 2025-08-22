from django.urls import path
from .views import FindOnePersonUploadView

urlpatterns = [
    path("find-one-upload/", FindOnePersonUploadView.as_view(), name="find_one_upload"),
]
