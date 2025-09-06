from django.db import models

class PersonClass(models.Model):
    id = models.BigAutoField(primary_key=True)
    file_name = models.CharField(max_length=512)
    person_name = models.CharField(max_length=255)

    class Meta:
        db_table = "base_image"
        managed = False

