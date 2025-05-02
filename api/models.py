# api/models.py
from django.db import models

class ImageRecord(models.Model):
    title       = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    image       = models.ImageField(upload_to="uploads/")
    created_at  = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
