import mimetypes
from django.conf import settings
from supabase import create_client


def _get_client():
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


def upload_image_to_supabase(file_obj, storage_path: str) -> str:
    """
    Upload an in-memory file to Supabase Storage.

    Parameters
    ----------
    file_obj : Django InMemoryUploadedFile / TemporaryUploadedFile
        The raw file from request.FILES.
    storage_path : str
        Destination path inside the bucket, e.g. "analysis_images/<uuid>/photo.jpg".

    Returns
    -------
    str
        The public URL of the uploaded file.

    Raises
    ------
    RuntimeError
        If SUPABASE_IMAGES_BUCKET is not configured.
    """
    bucket = settings.SUPABASE_IMAGES_BUCKET
    if not bucket:
        raise RuntimeError(
            "SUPABASE_IMAGES_BUCKET is not set. Add it to your .env file."
        )

    content_type, _ = mimetypes.guess_type(file_obj.name)
    content_type = content_type or "application/octet-stream"

    client = _get_client()
    data = file_obj.read()

    client.storage.from_(bucket).upload(
        storage_path,
        data,
        file_options={"content-type": content_type},
    )

    return f"{settings.SUPABASE_URL}/storage/v1/object/public/{bucket}/{storage_path}"
