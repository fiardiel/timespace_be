import json
import tempfile
from pathlib import Path

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser

from .serializers import FindOnePersonUploadSerializer
from FaceTrack import find_people_in_group_simple  # your function

def _save_uploaded_file(dst: Path, django_file) -> None:
    with dst.open("wb") as f:
        for chunk in django_file.chunks():
            f.write(chunk)

class FindOnePersonUploadView(APIView):
    """
    POST /api/facetrack/find-one-upload/
    multipart/form-data:
      - group_img: file (required)
      - person_img: file (required)
      - verbose: bool (optional)
    Returns: {status, found, total_faces}
    """
    parser_classes = [MultiPartParser, FormParser]
    authentication_classes = []   # add auth in prod
    permission_classes = []

    def post(self, request):
        ser = FindOnePersonUploadSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        verbose = ser.validated_data.get("verbose", False)

        try:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)

                # Save group image
                group_path = td_path / "group.jpg"
                _save_uploaded_file(group_path, ser.validated_data["group_img"])

                # Save single person image into a temp 'people' directory
                person_dir = td_path / "people"
                person_dir.mkdir(parents=True, exist_ok=True)
                person_path = person_dir / "person.jpg"
                _save_uploaded_file(person_path, ser.validated_data["person_img"])

                # Run your pipeline (expects group_img + person_directory)
                result_json = find_people_in_group_simple(
                    group_img=group_path,
                    person_directory=person_dir,
                    verbose=verbose,
                )
                return Response(json.loads(result_json))
        except Exception as e:
            return Response({"status": False, "error": str(e)},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
