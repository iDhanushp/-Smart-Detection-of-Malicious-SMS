from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request,'index.html',{})
def About(request):
    return render(request,'About.html',{})
def contact(request):
    return render(request,'contact.html',{})
def furniture(request):
    return render(request,'furniture.html',{})
def shop(request):
    return render(request,'shop.html',{})
