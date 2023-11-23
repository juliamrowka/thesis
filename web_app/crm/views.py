from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import SignUpForm
# from .models import Document
import openpyxl
import joblib

# Create your views here.

def home(request):
    # Check to see if logging in
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        # Authenticate
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, "You have been logged in!")
            return redirect('home')
        else:
            messages.success(request, "There was an error logging in. Please try again...")
            return redirect('home')
    else:
        return render(request, 'home.html', {})

def logout_user(request):
    logout(request)
    messages.success(request, "You have been logged out...")
    return redirect('home')

def register_user(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            # Authenticate and login
            username = form.cleaned_data['username']
            password = form.cleaned_data['password1']
            user = authenticate(username=username, password=password)
            login(request, user)
            messages.success(request, "You have successfully registered! Welcome!")
            return redirect('home')
    else:
        form = SignUpForm()
        return render(request, 'register.html', {'form':form})
    return render(request, 'register.html', {'form':form})

# form vine page
loaded_rf_model = joblib.load("ML_Model/rf_model.joblib")

def analysis(request):
	if request.method == 'POST':
        # These are all the variables that we obtained from the user through a POST Request. 
		alchohol_content=request.POST.get('alchohol_content','default')
		malic_acid=request.POST.get('malic_acid','default')
		Ash=request.POST.get('Ash','default')
		alc_ash=request.POST.get('alc_ash','default')
		Magnesium=request.POST.get('Magnesium','default')
		Phenols=request.POST.get('Phenols','default')
		Flavanoid=request.POST.get('Flavanoid','default')
		NFPhelons=request.POST.get('NFPhelons','default')
		Cyacnins=request.POST.get('Cyacnins','default')		
		Intensity=request.POST.get('Intensity','default')
		Hue=request.POST.get('Hue','default')
		OD280=request.POST.get('OD280','default')
		Proline=request.POST.get('Proline','default')

		labels=[[float(alchohol_content),
			float(malic_acid),
			float(Ash),
			float(alc_ash),
			float(Magnesium),
			float(Phenols),
			float(Flavanoid),
			float(NFPhelons),
			float(Cyacnins),
			float(Intensity),
			float(Hue),
			float(OD280),
			float(Proline)
		]]
		# Now, predicting the quality of wine based on our parameters.
		our_labels = loaded_rf_model.predict(labels)

		if our_labels[0]<=400:
			wine_quality="A Poor Quality Wine"
		if 400<our_labels[0]<=800:
			wine_quality="A Average Quality Wine"
		if 800<our_labels[0]<=1200:
			wine_quality="A Good Quality Wine"
		if 1200<our_labels[0]<=1500:
			wine_quality="A Exclusive Wine"
		if our_labels[0]>1500:
			wine_quality="A Premium & Fresh Wine"		

		details={
			"answer":our_labels[0],
			"wine_quality":wine_quality,
		}

		# Now rendering our results page with the data.	
		return render(request, "data_analysis/results.html", details)
    
	return render(request, "analysis.html")
