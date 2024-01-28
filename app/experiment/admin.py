from django.contrib import admin
from .models import Document, MLModel

class DocumentAdmin(admin.ModelAdmin):
    fieldsets = [
        (None, {"fields":["document"]}),
        (None, {"fields":["description"]}),
        # ("Date information", {"fields": ["uploaded_at"], "classes": ["collapse"]}),
    ]
    list_display = ["user_id", "document", "description", "uploaded_at"]

class MLModelAdmin(admin.ModelAdmin):
    fieldsets = [
        (None, {"fields":["file"]}),
        # (None, {"fields":["description"]}),
        # ("Date information", {"fields": ["uploaded_at"], "classes": ["collapse"]}),
    ]
    list_display = ["user_id", "file", "uploaded_at"]

# Register your models here.
admin.site.register(Document, DocumentAdmin)
admin.site.register(MLModel, MLModelAdmin)