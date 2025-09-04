# facetrack/views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .utils import download_base_images

@api_view(["POST"])
def fetch_base_images(request):
    files = download_base_images(clear_first=True)  # always clear then download
    return Response({"cleared": True, "downloaded": files, "count": len(files)})
