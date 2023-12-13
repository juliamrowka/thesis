from django.shortcuts import render, redirect, get_object_or_404
# from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import DocumentForm
from .models import Document
import openpyxl as op
# from io import BytesIO
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import pickle
# import joblib
# Create your views here.

def model_form_upload(request):
    """
    show upload form and when the file is uploaded preview this file
    """
    if request.user.is_authenticated:
        if request.method == 'POST':
            form = DocumentForm(request.POST, request.FILES)
            filename = "documents/user_" + str(request.user.id) + "/" + str(request.FILES['document'])
            files = Document.objects.filter(document = filename, user = request.user)
            if form.is_valid() and files.count() == 0:
                newdoc = Document(document = request.FILES['document'], description = request.POST['description'])
                newdoc.user = request.user
                newdoc.save()
                messages.success(request, "You have successfully uploaded file!")
                request.session['filename'] = filename
                
                return redirect('experiment')
            elif files.count() > 0:
                messages.success(request, "A file with that name already exists.")
        else:
            form = DocumentForm()
            # return render(request, 'upload.html', {'form': form})
        documents = Document.objects.filter(user=request.user)
        return render(request, 'upload.html', {'documents': documents, 'form': form})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')

def experiment(request):
      if request.user.is_authenticated:
        if 'filename' in request.session:
            excel_name = str(request.session['filename'])
            excel_data = list()
            wb = op.load_workbook(request.session['filename'])
            worksheet = wb.active
            print(worksheet)
            # iterating over the rows and
            # getting value from each cell in row
            for row in worksheet.iter_rows():
                row_data = list()
                for cell in row:
                    row_data.append(str(cell.value))
                excel_data.append(row_data)
            return render(request, 'experiment.html', {'excel_data': excel_data, 'excel_name': excel_name})
        else:
            messages.success(request, "You need to choose or upload a file first")
            return redirect('upload') 
      else:
        messages.success(request, "You need to log in first")
        return redirect('home')
      
# def choose_file(request, file_id):
#     filename = get_object_or_404(Document, pk=file_id)
#     print(filename)
#     request.session['filename'] = filename
#     return redirect('home')

my_dict = {'StandardScaler': StandardScaler(), 'MinMaxScaler': MinMaxScaler()}

def transformer(request):
    if request.user.is_authenticated:
        return render(request, 'transformer.html', {})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')

def std(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            column = request.POST['choosen_column']
            if 'my_pipeline' not in request.session:
                request.session['my_pipeline'] = ['StandardScaler', int(column)]
                print('just have created pipeline')
            else:
                pipe = request.session['my_pipeline']
                pipe.append(['StandardScaler', int(column)])
                request.session['my_pipeline'] = pipe

                print(f'pipe exists: {pipe}')
                print(f'pipeline in session {request.session["my_pipeline"]}')
            return redirect('transformer')

        return render(request, 'std.html', {})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')
    
def minmax(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            column = request.POST['choosen_column']
            if 'my_pipeline' not in request.session:
                request.session['my_pipeline'] = ['MinMaxScaler', int(column)]
                print('no pipeline')
            else:
                pipe = request.session['my_pipeline']
                pipe.append(['MinMaxScaler', int(column)])
                request.session['my_pipeline'] = pipe

                print(f'pipe exists: {pipe}')
                print(f'pipeline in session {request.session["my_pipeline"]}')
            return redirect('transformer')
            

        return render(request, 'minmax.html', {})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')
    
def norm(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            norm_type = request.POST['norm_type']
            column = request.POST['choosen_column']
            if 'my_pipeline' not in request.session:
                request.session['my_pipeline'] = ['Normalizer', norm_type, int(column)]
                print('no pipeline')
            else:
                pipe = request.session['my_pipeline']
                pipe.append(['Normalizer', norm_type, int(column)])
                request.session['my_pipeline'] = pipe

                print(f'pipe exists: {pipe}')
                print(f'pipeline in session {request.session["my_pipeline"]}')
            return redirect('transformer')
            
        return render(request, 'norm.html', {})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')
    
def pca(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            parameter_n = request.POST['parameter_n']
            if 'my_pipeline' not in request.session:
                request.session['my_pipeline'] = ['PCA', int(parameter_n)]
                print('no pipeline')
            else:
                pipe = request.session['my_pipeline']
                pipe.append(['PCA', int(parameter_n)])
                request.session['my_pipeline'] = pipe

                print(f'pipe exists: {pipe}')
                print(f'pipeline in session {request.session["my_pipeline"]}')
            return redirect('transformer')
            
        return render(request, 'pca.html', {})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')