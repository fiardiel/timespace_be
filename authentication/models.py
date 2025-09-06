from django.db import models

class Imagerecord(models.Model):
    file_name = models.TextField()
    event = models.TextField()
    people = models.JSONField()

    class Meta:
        managed = False
        db_table = 'imagerecord'

    @property
    def image_url(self):
        prefix = "https://tbymrebzmgbidchigdlc.supabase.co/storage/v1/object/public/timespace//"
        return f"{prefix}{self.file_name}"
