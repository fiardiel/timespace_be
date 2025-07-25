from django.shortcuts import render
from .models import Imagerecord
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions, status
from authentication.serializers import UserSerializer


def imagerecord_list(request):
    records = Imagerecord.objects.all()
    return render(request, 'imagerecorjd_list.html', {'records': records})


class Register(APIView):
    permission_classes = [permissions.IsAdminUser]

    def post(self, request):
            serializer = UserSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            serializer.save()
            return Response(serializer.data, status=201)
