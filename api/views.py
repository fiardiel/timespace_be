# api/views.py
from rest_framework import viewsets
from .models import ImageRecord
from .serializers import ImageRecordSerializer

class ImageRecordViewSet(viewsets.ModelViewSet):
    queryset         = ImageRecord.objects.all()
    serializer_class = ImageRecordSerializer

