# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render, HttpResponse

# Create your views here.

def test(request):
    aa = 'hello word !'
    return render(request, 'home.html', {'ff': aa})
    return HttpResponse(aa)


