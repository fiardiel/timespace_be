from pathlib import Path
import json

from rest_framework.decorators import api_view
from rest_framework.response import Response

from django.conf import settings

from .utils import download_base_images, first_image_in_dir, upsert_latest_imagerecord_people


@api_view(["POST"])
def fetch_base_images(request):
    """
    Clears & downloads base images and the latest group photo,
    then runs the ML pipeline and returns combined JSON.
    """
    # 1) Download both folders (your utils already pulls the group photo too)
    files = download_base_images(clear_first=True)

    # 2) Locate the inputs for ML
    base_dir   = Path(settings.BASE_DIR) / "facetrack"
    group_dir  = base_dir / "GroupPhotoFromFE"
    person_dir = base_dir / "PeopleFromDataBase"

    group_img = first_image_in_dir(group_dir)
    if group_img is None:
        return Response({
            "cleared": True,
            "downloaded": files,
            "count": len(files),
            "ml": {
                "status": False,
                "message": f"No group photo found in {group_dir}. Did the download step succeed?"
            }
        }, status=200)

    # 3) Lazy-import the ML function (so model loads only when the endpoint is hit)
    try:
        from .FaceTrack import find_people_in_group_simple
    except Exception as e:
        return Response({
            "cleared": True,
            "downloaded": files,
            "count": len(files),
            "ml": {
                "status": False,
                "message": "Failed to import ML module",
                "error": str(e)
            }
        }, status=500)

    # 4) Run the pipeline
    try:
        result_json = find_people_in_group_simple(group_img, person_dir, verbose=False)
        ml_result = json.loads(result_json)
    except Exception as e:
        return Response({
            "cleared": True,
            "downloaded": files,
            "count": len(files),
            "group_img": str(group_img),
            "ml": {
                "status": False,
                "message": "ML inference failed",
                "error": str(e)
            }
        }, status=500)

    # 4.5) Rewrite `people` in the latest imagerecord using ML results
    try:
        found_list = ml_result.get("found", []) if isinstance(ml_result, dict) else []
        db_write = upsert_latest_imagerecord_people(found_list)
    except Exception as e:
        db_write = {"updated": False, "error": str(e)}


    # 5) Success payload
    return Response({
        "cleared": True,
        "downloaded": files,
        "count": len(files),
        "group_img": str(group_img),
        "ml": ml_result
    }, status=200)
