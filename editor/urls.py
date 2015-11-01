# -*-*- encoding: utf-8 -*-*-
# pylint: disable=E1101
from __future__ import absolute_import, unicode_literals
from django.conf.urls import url
from .views import (
    NetworkDefinitionList, NetworkDefinitionEdit,
    NetworkDefinitionCreate, NetworkDefinitionDelete,
    NetworkParameterEdit)

urlpatterns = [
    url(r'^$', NetworkDefinitionList.as_view(), name='definition_list'),
    url(r'^new/$', NetworkDefinitionCreate.as_view(), name='definition_create'),
    url(r'^(?P<pk>[0-9]+)/$', NetworkDefinitionEdit.as_view(), name='definition_edit'),
    url(r'^(?P<pk>[0-9]+)/parameters/$', NetworkParameterEdit.as_view(), name='parameter_edit'),
    url(r'^(?P<pk>[0-9]+)/delete/$', NetworkDefinitionDelete.as_view(), name='definition_delete')
]
