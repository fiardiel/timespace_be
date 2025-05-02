# api/views.py
from rest_framework.viewsets import ModelViewSet
from api.models                   import ImageRecord
from api.serializers              import ImageRecordSerializer

class ImageRecordViewSet(ModelViewSet):
    queryset         = ImageRecord.objects.all()
    serializer_class = ImageRecordSerializer
