from django.shortcuts import render, redirect, get_object_or_404
# from django.contrib.auth import authenticate, login, logout
# from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import DocumentForm
from .models import Document
import openpyxl as op
import os
import pandas as pd
# from sklearn import preprocessing
# from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

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
                # request.session['my_pipeline'] = []
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
        if 'my_pipeline' in request.session: del request.session['my_pipeline']
        if 'my_estimator' in request.session: del request.session['my_estimator']
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
            del request.session['filename']
            if 'max_column' in request.session: del request.session['max_column']
            if 'my_pipeline' in request.session: del request.session['my_pipeline']
            if 'my_estimator' in request.session: del request.session['my_estimator']
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
                my_pipeline = None
                my_estimator = None
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
                if 'my_estimator' in request.session:
                    my_estimator = request.session['my_estimator'] 
                return render(request, 'experiment.html', {'excel_data': excel_data, 'excel_name': excel_name, 'my_pipeline': my_pipeline, 'my_estimator': my_estimator})
                # else:
                #     return render(request, 'experiment.html', {'excel_data': excel_data, 'excel_name': excel_name})
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
                request.session['my_pipeline'] = [['StandardScaler', ('Chosen column: ', int(column))]]
                # print('just have created pipeline')
            else:
                pipe = request.session['my_pipeline']
                pipe.append(['StandardScaler', ('Chosen column: ', int(column))])
                request.session['my_pipeline'] = pipe

                # print(f'pipe exists: {pipe}')
                # print(f'pipeline in session {request.session["my_pipeline"]}')
            return redirect('transformer')
        elif 'max_column' in request.session:
            max_column = request.session['max_column']
            # print(type(max_column))
            return render(request, 'preprocessing/std.html', {'max_column': max_column})
        else:
            messages.success(request, "You need to choose data file first")
        return redirect('upload')
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
                request.session['my_pipeline'] = [['MinMaxScaler', ('Chosen column: ', int(column))]]
                # print('no pipeline')
            else:
                pipe = request.session['my_pipeline']
                pipe.append(['MinMaxScaler', ('Chosen column: ', int(column))])
                request.session['my_pipeline'] = pipe

                # print(f'pipe exists: {pipe}')
                # print(f'pipeline in session {request.session["my_pipeline"]}')
            return redirect('transformer')
        elif 'max_column' in request.session:
            max_column = request.session['max_column']
            # print(type(max_column))
            return render(request, 'preprocessing/minmax.html', {'max_column': max_column})
        else:
            messages.success(request, "You need to choose data file first")
        return redirect('upload')
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
            if 'my_pipeline' not in request.session:
                request.session['my_pipeline'] = [['Normalizer', ('Norm type: ', norm_type)]]
                # print('no pipeline')
            else:
                pipe = request.session['my_pipeline']
                pipe.append(['Normalizer', ('Norm type: ', norm_type)])
                request.session['my_pipeline'] = pipe

                # print(f'pipe exists: {pipe}')
                # print(f'pipeline in session {request.session["my_pipeline"]}')
            return redirect('transformer')
        elif 'max_column' in request.session:
            return render(request, 'preprocessing/norm.html', {})
        else:
            messages.success(request, "You need to choose data file first")
        return redirect('upload')
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
                request.session['my_pipeline'] = [['PCA', ('Chosen n parameter: ', int(parameter_n))]]
                # print('no pipeline')
            else:
                pipe = request.session['my_pipeline']
                pipe.append(['PCA', ('Chosen n parameter: ', int(parameter_n))])
                request.session['my_pipeline'] = pipe

                # print(f'pipe exists: {pipe}')
                # print(f'pipeline in session {request.session["my_pipeline"]}')
            return redirect('transformer')
        elif 'max_column' in request.session:
            n_component = request.session['max_column']
            return render(request, 'preprocessing/pca.html', {'n_component': n_component})
        else:
            messages.success(request, "You need to choose data file first")
        return redirect('upload')
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')
    
def estimator(request):
    """
    description
    """
    if request.user.is_authenticated:
        return render(request, 'estimator.html', {})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')
    
def ord_least_squares(request):
    """
    description
    """
    # if request.user.is_authenticated:
    #     return render(request, 'ordinary-least-squares.html', {})
    # else:
    #     messages.success(request, "You need to log in first")
    #     return redirect('home')
    if request.user.is_authenticated:
        if request.method == 'POST':
            column = request.POST['choosen_column']
            request.session['my_estimator'] = ['LinearRegression', [('Column of target values: ', int(column))]]
            return redirect('experiment')
        else:
            return render(request, 'regression/ordinary-least-squares.html', {'max_column': request.session['max_column']})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')
    
def svm_regression(request):
    """
    description
    """
    if request.user.is_authenticated:
        if request.method == 'POST':
            epsilon = request.POST['epsilon']
            C_parameter = request.POST['C_parameter']
            intercept_scaling = request.POST['intercept_scaling']
            column = request.POST['choosen_column']
            request.session['my_estimator'] = ['LinearSVR', [('Epsilon: ', epsilon), ('Regularization parameter C: ', C_parameter), ('Intercept scaling: ', intercept_scaling), ('Column of target values: ', int(column))]]
            return redirect('experiment')
        else:
            return render(request, 'regression/svm-regression.html', {'max_column': request.session['max_column']})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')
    
def nn_regression(request):
    """
    description
    """
    if request.user.is_authenticated:
        if request.method == 'POST':
            neighbors = request.POST['neighbors']
            # weight = request.POST['weight']
            # p_parameter = request.POST['p_parameter']
            column = request.POST['choosen_column']
            request.session['my_estimator'] = ['KNeighborsRegressor', [('Number of neighbors: ', neighbors), ('Column of target values: ', int(column))]]
            return redirect('experiment')
        else:
            return render(request, 'regression/nn-regression.html', {'max_column': request.session['max_column']})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')
    
def dt_regression(request):
    """
    description
    """
    if request.user.is_authenticated:
        if request.method == 'POST':
            criterion = request.POST['criterion']
            max_leaf_nodes = request.POST['max_leaf_nodes']
            column = request.POST['choosen_column']
            request.session['my_estimator'] = ['DecisionTreeRegressor', [('Criterion: ', criterion), ('Max leaf nodes: ', max_leaf_nodes), ('Column of target values: ', int(column))]]
            return redirect('experiment')
        else:
            return render(request, 'regression/dt_regression.html', {'max_column': request.session['max_column']})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')
    
def categorical_nb(request):
    """
    description
    """
    if request.user.is_authenticated:
        if request.method == 'POST':
            alpha = request.POST['alpha']
            column = request.POST['choosen_column']
            request.session['my_estimator'] = ['CategoricalNB', [('Alpha: ', alpha), ('Column of target values: ', int(column))]]
            return redirect('experiment')
        else:
            return render(request, 'classification/categorical_nb.html', {'max_column': request.session['max_column']})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')
    
def svm_classification(request):
    """
    description
    """
    if request.user.is_authenticated:
        if request.method == 'POST':
            c_parameter = request.POST['c_parameter']
            class_weight = request.POST['class_weight']
            column = request.POST['choosen_column']
            request.session['my_estimator'] = ['LinearSVC', [('C parameter: ', c_parameter), ('Class weight: ', class_weight), ('Column of target values: ', int(column))]]
            return redirect('experiment')
        else:
            return render(request, 'classification/svm_classification.html', {'max_column': request.session['max_column']})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')
    
def nn_classification(request):
    """
    description
    """
    if request.user.is_authenticated:
        if request.method == 'POST':
            neighbors = request.POST['neighbors']
            # weight = request.POST['weight']
            # p_parameter = request.POST['p_parameter']
            column = request.POST['choosen_column']
            request.session['my_estimator'] = ['KNeighborsClassifier', [('Number of neighbors: ', neighbors), ('Column of target values: ', int(column))]]
            return redirect('experiment')
        else:
            return render(request, 'classification/nn-classification.html', {'max_column': request.session['max_column']})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')

def dt_classification(request):
    """
    description
    """
    if request.user.is_authenticated:
        if request.method == 'POST':
            criterion = request.POST['criterion']
            max_leaf_nodes = request.POST['max_leaf_nodes']
            column = request.POST['choosen_column']
            request.session['my_estimator'] = ['DecisionTreeClassifier', [('Criterion: ', criterion), ('Max leaf nodes: ', max_leaf_nodes), ('Column of target values: ', int(column))]]
            return redirect('experiment')
        else:
            return render(request, 'classification/dt_classification.html', {'max_column': request.session['max_column']})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')
    
def compute(request):
    """
    description
    """
    if request.user.is_authenticated:
        if 'my_pipeline' and 'my_estimator' in request.session:
            transformer_list = []
            for i in request.session['my_pipeline']:
                if i[0] == 'Normalizer': transformer_list.append(('norm', Normalizer(norm=i[1][1])))
                if i[0] == 'PCA': transformer_list.append(('pca', PCA(n_components=i[1][1])))
                # pass
            print(transformer_list)

            # header_column = 0
            # cols = "A:B"
            wb = pd.read_excel(io=request.session['filename'], header=0)
            print(wb)
            # print(type(wb))
            ct= ColumnTransformer([('std',StandardScaler(),[0]), ('std2',StandardScaler(),[1])], remainder="passthrough")
            ct.fit(wb)
            res_ct=ct.transform(wb)
            print(res_ct)
            # print(wb.iloc[:, 0])
            # print(wb.iloc[:, 1:4])

            # print(request.session['my_estimator'][len(request.session['my_estimator'])-1][1])
            target_column = request.session['my_estimator'][1][len(request.session['my_estimator'][1])-1][1] - 1
            print(f'target_column: {target_column}')
            # # target = wb.iloc[:, target_column-1]
            # print(type(request.session['my_estimator']))

            X_train,X_test,y_train,y_test=train_test_split(wb.iloc[:,[x for x in range(len(wb.columns)) if x!=target_column]], wb.iloc[:, target_column],test_size=0.2,random_state=0)
            # X_train,X_test,y_train,y_test=train_test_split(wb.iloc[:, 1:4],wb.iloc[:, 0],test_size=0.2,random_state=0)
            # iris = datasets.load_iris()
            # X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=0)
            pipe4=Pipeline([('pca', PCA(n_components=2)),('tree', DecisionTreeClassifier())])

            pipe4.fit(X_train,y_train)
            res=pipe4.predict(X_train)
            print(np.transpose(np.array([y_train,res])))
            print(pipe4.score(X_test,y_test))
            scores=cross_val_score(pipe4,X_train,y_train,cv=10)
            print(f"Średnia {scores.mean()}, odchylenie standardowe {scores.std()}")
            messages.success(request, f"Średnia {scores.mean()}, odchylenie standardowe {scores.std()}")
            return render(request, 'compute.html', {})
        else:
            messages.success(request, "No data")
            return render(request, 'compute.html', {}) 
        # return render(request, 'compute.html', {})
    else:
        messages.success(request, "You need to log in first")
        return redirect('home')