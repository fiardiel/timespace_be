import os
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

import requests
from django.conf import settings
from django.db import connection


# allow clearing either folder safely
ALLOWED_CLEAR_DIRS = {"PeopleFromDataBase", "GroupPhotoFromFE"}


def _safe_empty_dir(path: Path):
    """
    Empties `path` but keeps the directory itself.
    Includes safety guards to avoid accidental deletion elsewhere.
    """
    path.mkdir(parents=True, exist_ok=True)

    # Safety checks — refuse to clear unexpected folders.
    if path.name not in ALLOWED_CLEAR_DIRS:
        raise RuntimeError(f"Refusing to clear non-target folder: {path.name}")
    project_root = Path(settings.BASE_DIR).resolve()
    if project_root not in path.resolve().parents:
        raise RuntimeError("Refusing to clear a directory outside the project.")

    # Remove all files/subfolders inside target
    for p in path.iterdir():
        if p.is_file() or p.is_symlink():
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        elif p.is_dir():
            shutil.rmtree(p)


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", (name or "").lower()).strip("_") or "file"


def download_last_group_photo(clear_first: bool = True):
    """
    Fetch the last (highest id) image from public.imagerecord
    and download its file_name into facetrack/GroupPhotoFromFE.
    """
    with connection.cursor() as cur:
        cur.execute("""
            SELECT id, file_name, event
            FROM public.imagerecord
            ORDER BY id DESC
            LIMIT 1;
        """)
        row = cur.fetchone()

    if not row:
        return {"saved": False, "reason": "No rows in public.imagerecord"}

    rec_id, file_url, event = row

    dest_dir = Path(settings.BASE_DIR) / "facetrack" / "GroupPhotoFromFE"
    if clear_first:
        _safe_empty_dir(dest_dir)
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(file_url or "")
    basename = os.path.basename(parsed.path) or f"group_{rec_id}"
    _, ext = os.path.splitext(basename)
    if not ext:
        ext = ".jpg"

    filename = f"{_slugify(event or 'group')}_{rec_id}{ext}"
    dest_path = dest_dir / filename

    r = requests.get(file_url, stream=True, timeout=30)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(1024 * 32):
            if chunk:
                f.write(chunk)

    return {"saved": True, "saved_to": str(dest_path), "id": rec_id, "event": event}


def download_base_images(clear_first: bool = True, also_download_group: bool = True):
    """
    Clears facetrack/PeopleFromDataBase (if clear_first),
    downloads all images from public.base_image into it,
    then (optionally) clears & downloads the last group photo into GroupPhotoFromFE.
    Returns the list of downloaded base image file paths (unchanged API).
    """
    save_dir = Path(settings.BASE_DIR) / "facetrack" / "PeopleFromDataBase"

    if clear_first:
        _safe_empty_dir(save_dir)
    else:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Fetch URLs + names from DB
    with connection.cursor() as cur:
        cur.execute("SELECT file_name, person_name FROM public.base_image;")
        rows = cur.fetchall()

    downloaded = []
    for file_url, person_name in rows:
        ext = os.path.splitext(urlparse(file_url).path)[1] or ".jpg"
        fname = f"{_slugify(person_name)}{ext}"
        dest = save_dir / fname

        r = requests.get(file_url, stream=True, timeout=30)
        r.raise_for_status()

        with open(dest, "wb") as f:
            for chunk in r.iter_content(1024 * 32):
                if chunk:
                    f.write(chunk)

        downloaded.append(str(dest))

    # NEW: also pull the latest group photo
    if also_download_group:
        try:
            download_last_group_photo(clear_first=True)
        except Exception as e:
            # keep base downloads working even if group photo fails
            # you can log this if you have logging configured
            pass

    return downloaded
