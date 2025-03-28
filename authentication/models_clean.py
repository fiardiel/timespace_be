from django.db import models

class Imagerecord(models.Model):
    file_name = models.TextField()
    event = models.TextField()
    people = models.JSONField()

    class Meta:
        managed = False
        db_table = 'imagerecord'