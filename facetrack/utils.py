import os
import requests
from django.conf import settings
from django.db import connection

def download_base_images():
    save_dir = os.path.join(settings.BASE_DIR, "facetrack", "PeopleFromDataBase")
    os.makedirs(save_dir, exist_ok=True)

    with connection.cursor() as cur:
        cur.execute("SELECT file_name, person_name FROM public.base_image;")
        rows = cur.fetchall()

    downloaded = []
    for file_url, person_name in rows:
        file_ext = os.path.splitext(file_url)[1] or ".jpg"
        save_path = os.path.join(save_dir, f"{person_name}{file_ext}")

        r = requests.get(file_url, stream=True)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            downloaded.append(save_path)

    return downloaded
