# -*-*- encoding: utf-8 -*-*-
from __future__ import absolute_import, unicode_literals
from django.conf.urls import include, url
# from django.contrib import admin
from .views import Home


urlpatterns = [
    # url(r'^admin/', include(admin.site.urls)),
    url(r'^editor/', include('editor.urls')),
    url(r'^$', Home.as_view(), name='home'),
]
