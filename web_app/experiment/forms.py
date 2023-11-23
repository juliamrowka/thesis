from django import forms
from .models import Document

class DocumentForm(forms.ModelForm):
    description = forms.CharField(label="", widget=forms.TextInput(attrs={'class':'form-control', 'placeholder':'Description'}))
    document = forms.FileField(label="", widget=forms.FileInput(attrs={'class':'form-control'}))    
    class Meta:
        model = Document
        fields = ('description', 'document')