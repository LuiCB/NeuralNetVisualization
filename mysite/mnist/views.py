from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render
from mnist.models import InputForm
import os
cwd = os.getcwd()
print(cwd)
import NeuralNet.testNN as nn
from multiprocessing import Process, Pool
p = Pool(1)

def index(request):
    global p 
    template = loader.get_template('mnist.html')
    context = {}
    print(request.method)
    if 'start' in request.POST:
        print("start")
        p.apply_async(nn.main)
    print("here")
    if 'stop' in request.POST:
        print('stop')
        if p != None:
            p.terminate()
            p = Pool(1)
    return HttpResponse(template.render(context, request))
