# -*-*- encoding: utf-8 -*-*-
from __future__ import absolute_import, unicode_literals
from django.views.generic import TemplateView
from editor.models import NetworkDefinition


class Home(TemplateView):
    """
    List of all network definitions.
    """
    template_name = "index.html"

    def get_context_data(self, **kwargs):
        context = super(Home, self).get_context_data(**kwargs)
        context['object_list'] = list(NetworkDefinition.objects.all())
        return context
