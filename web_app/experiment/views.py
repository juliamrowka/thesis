from django.shortcuts import render, redirect, get_object_or_404
# from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import DocumentForm
from .models import Document
import openpyxl as op
import os
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
    
def choose_file(request, pk):
    """
    description
    """
    if request.user.is_authenticated:
        choosen_file = Document.objects.get(id=pk)
        filename = choosen_file.document
        messages.success(request, "You have successfully chosen file!")
        request.session['filename'] = str(filename)
        request.session['my_pipeline'] = []
        # print(filename)
        return redirect('experiment')
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')
    
def delete_file(request, pk):
    """
    description
    """
    if request.user.is_authenticated:
        to_delete = get_object_or_404(Document, pk=pk)
        print(f'File to delete: {to_delete}')
        filename = str(to_delete.document)
        print(f'Path of file to delete: {filename}')
        if to_delete and filename:
            to_delete.delete()
            os.remove(filename)
            print('file removed')
            messages.success(request, "You have successfully removed file!")
        if filename == request.session['filename']:
            request.session['filename'] = ''
            request.session['max_column'] = ''
            request.session['my_pipeline'] = []
            print('session filename')
        return redirect('experiment')
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')

def experiment(request):
    """
    description
    """
    if request.user.is_authenticated:
        if 'filename' in request.session:
            if len(request.session['filename']) > 0:
                excel_name = str(request.session['filename'])
                excel_data = list()
                wb = op.load_workbook(request.session['filename'])
                worksheet = wb.active
                request.session['max_column'] = list(range(1,worksheet.max_column + 1))
                for row in worksheet.iter_rows():
                    row_data = list()
                    for cell in row:
                        row_data.append(str(cell.value))
                    excel_data.append(row_data)
                if 'my_pipeline' in request.session:
                    my_pipeline = request.session['my_pipeline']
                    return render(request, 'experiment.html', {'excel_data': excel_data, 'excel_name': excel_name, 'my_pipeline': my_pipeline})
                else:
                    return render(request, 'experiment.html', {'excel_data': excel_data, 'excel_name': excel_name})
            else:
                return render(request, 'experiment.html', {})
        else:
            return render(request, 'experiment.html', {})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')

def transformer(request):
    """
    description
    """
    if request.user.is_authenticated:
        return render(request, 'transformer.html', {})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')

def std(request):
    """
    description
    """
    if request.user.is_authenticated:
        if request.method == 'POST':
            column = request.POST['choosen_column']
            if 'my_pipeline' not in request.session:
                request.session['my_pipeline'] = [['StandardScaler', int(column)]]
                print('just have created pipeline')
            else:
                pipe = request.session['my_pipeline']
                pipe.append(['StandardScaler', int(column)])
                request.session['my_pipeline'] = pipe

                print(f'pipe exists: {pipe}')
                print(f'pipeline in session {request.session["my_pipeline"]}')
            return redirect('transformer')
        else:
            max_column = request.session['max_column']
            print(type(max_column))
            return render(request, 'std.html', {'max_column': max_column})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')
    
def minmax(request):
    """
    description
    """
    if request.user.is_authenticated:
        if request.method == 'POST':
            column = request.POST['choosen_column']
            if 'my_pipeline' not in request.session:
                request.session['my_pipeline'] = [['MinMaxScaler', int(column)]]
                print('no pipeline')
            else:
                pipe = request.session['my_pipeline']
                pipe.append(['MinMaxScaler', int(column)])
                request.session['my_pipeline'] = pipe

                print(f'pipe exists: {pipe}')
                print(f'pipeline in session {request.session["my_pipeline"]}')
            return redirect('transformer')
        else:
            max_column = request.session['max_column']
            print(type(max_column))
            return render(request, 'minmax.html', {'max_column': max_column})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')
    
def norm(request):
    """
    description
    """
    if request.user.is_authenticated:
        if request.method == 'POST':
            norm_type = request.POST['norm_type']
            column = request.POST['choosen_column']
            if 'my_pipeline' not in request.session:
                request.session['my_pipeline'] = [['Normalizer', norm_type, int(column)]]
                print('no pipeline')
            else:
                pipe = request.session['my_pipeline']
                pipe.append(['Normalizer', norm_type, int(column)])
                request.session['my_pipeline'] = pipe

                print(f'pipe exists: {pipe}')
                print(f'pipeline in session {request.session["my_pipeline"]}')
            return redirect('transformer')
        else:
            max_column = request.session['max_column']
            print(type(max_column))
            return render(request, 'norm.html', {'max_column': max_column})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')
    
def pca(request):
    """
    description
    """
    if request.user.is_authenticated:
        if request.method == 'POST':
            parameter_n = request.POST['parameter_n']
            if 'my_pipeline' not in request.session:
                request.session['my_pipeline'] = [['PCA', int(parameter_n)]]
                print('no pipeline')
            else:
                pipe = request.session['my_pipeline']
                pipe.append(['PCA', int(parameter_n)])
                request.session['my_pipeline'] = pipe

                print(f'pipe exists: {pipe}')
                print(f'pipeline in session {request.session["my_pipeline"]}')
            return redirect('transformer')
        else:
            n_component = request.session['max_column']
            print(type(n_component))
            return render(request, 'pca.html', {'n_component': n_component})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')