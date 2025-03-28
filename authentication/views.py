from django.shortcuts import render
from .models import Imagerecord

def imagerecord_list(request):
    records = Imagerecord.objects.all()
    return render(request, 'imagerecord_list.html', {'records': records})
