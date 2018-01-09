from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render
import os

def index(request):
    template = loader.get_template('welcomePage.html')
    if (request.method == 'POST'):
        print(request.POST)
    return HttpResponse(template.render())
