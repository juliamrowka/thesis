from django.shortcuts import render, redirect
# from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import DocumentForm
from .models import Document
import io
import openpyxl
import pandas as pd
# import joblib
# Create your views here.

def model_form_upload(request):
    """
    show upload form and when the file is uploaded preview this file
    """
    if request.user.is_authenticated:
        excel_data = list()
        if request.method == 'POST':
            form = DocumentForm(request.POST, request.FILES)
            filename = "documents/user_" + str(request.user.id) + "/" + str(request.FILES['document'])
            files = Document.objects.filter(document = filename, user = request.user)
            if form.is_valid() and files.count() == 0:
                newdoc = Document(document = request.FILES['document'], description = request.POST['description'])
                newdoc.user = request.user
                newdoc.save()
                messages.success(request, "You have successfully uploaded file!")
                excel_file = request.FILES['document']
                print(excel_file)
                wb = openpyxl.load_workbook(excel_file)
                worksheet = wb.active
                print(worksheet)
                # iterating over the rows and
                # getting value from each cell in row
                for row in worksheet.iter_rows():
                    row_data = list()
                    for cell in row:
                        row_data.append(str(cell.value))
                    excel_data.append(row_data)
                # return redirect('upload')
            elif files.count() > 0:
                messages.success(request, "A file with that name already exists.")
        else:
            form = DocumentForm()
            # return render(request, 'upload.html', {'form': form})
        documents = Document.objects.filter(user=request.user)
        return render(request, 'upload.html', {'documents': documents, 'form': form, 'excel_data': excel_data})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')
    
# def preview_file(request):
#     if request.user.is_authenticated:
#         excel_data = list()
#         print(excel_file)
#         if request.FILE:
#             excel_file = request.FILES['document']
#             # print(excel_file)
#             # data = pd.read_excel(io=excel_file)
#             # print(data)
#             wb = openpyxl.load_workbook(excel_file)
#             worksheet = wb.active
#             print(worksheet)
#             # iterating over the rows and
#             # getting value from each cell in row
#             for row in worksheet.iter_rows():
#                 row_data = list()
#                 for cell in row:
#                     row_data.append(str(cell.value))
#                 excel_data.append(row_data)
#             return render(request, 'upload.html', {'excel_data': excel_data})
#         else:
#             messages.success(request, "no file uploaded")
#             return redirect('upload')
#     else:
#         messages.success(request, "You need to log in first")
#         return redirect('home') 

# def model_form_upload(request):
#     # if request.user is not None:
#         if request.method == 'POST':
#             form = DocumentForm(request.POST, request.FILES)
#             if form.is_valid():
#                 newdoc = Document(document = request.FILES['document'], description = request.POST['description'])
#                 newdoc.user = request.user
#                 newdoc.save()
#                 messages.success(request, "You have successfully uploaded file!")
#                 return redirect('upload')
#             # excel_file = request.FILES["excel_file"]
#             # wb = openpyxl.load_workbook(excel_file)
#             # worksheet = wb ["Sheet1"]
#             # print(worksheet)
#             # excel_data = list()
#             # # iterating over the rows and
#             # # getting value from each cell in row
#             # for row in worksheet.iter_rows():
#             #     row_data = list()
#             #     for cell in row:
#             #         row_data.append(str(cell.value))
#             #     excel_data.append(row_data)
#             # return render(request, 'upload.html', {'excel_data': excel_data})
#         else:
#             form = DocumentForm()
#             # return render(request, 'upload.html', {'form': form})
#         documents = Document.objects.filter(user=request.user)
#         return render(request, 'upload.html', {'documents': documents, 'form': form})


def experiment(request):
      if request.user.is_authenticated:
        user_name = request.user.first_name
        return render(request, 'experiment.html', {})
      else:
        messages.success(request, "You need to log in first")
        return redirect('home')