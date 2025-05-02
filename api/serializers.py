# api/serializers.py
from rest_framework import serializers
from api.models      import ImageRecord

class ImageRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model  = ImageRecord
        fields = "__all__"
