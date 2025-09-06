from rest_framework import serializers

class FindOnePersonUploadSerializer(serializers.Serializer):
    group_img = serializers.ImageField(required=True)
    person_img = serializers.ImageField(required=True)
    verbose = serializers.BooleanField(required=False, default=False)
