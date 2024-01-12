from django.db import models
from django.contrib.auth.models import User

# Create your models here.
def user_directory_path(instance, filename):
    # file will be uploaded to MEDIA_ROOT/user_<id>/<filename>
    return f'documents/user_{instance.user.id}/data/{filename}'

def user_directory_path_models(instance, filename):
    # file will be uploaded to MEDIA_ROOT/user_<id>/<filename>
    return f'documents/user_{instance.user.id}/models/{filename}'

class Document(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    description = models.CharField(max_length=255, blank=True)
    document = models.FileField(upload_to=user_directory_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)

class MLModel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    # description = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to=user_directory_path_models)
    uploaded_at = models.DateTimeField(auto_now_add=True)