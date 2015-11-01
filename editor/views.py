# -*-*- encoding: utf-8 -*-*-
# pylint: disable=E1101,W0201
from __future__ import absolute_import, unicode_literals
import logging
from django.utils.decorators import method_decorator
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import ListView, UpdateView, CreateView, View
from django.core.urlresolvers import reverse_lazy
from .models import NetworkDefinition
from .forms import NetworkDefinitionForm, NetworkDefinitionParameterForm, ParameterFormSet

logger = logging.getLogger(__name__)


class NetworkDefinitionList(ListView):
    """
    List of all network definitions.
    """
    model = NetworkDefinition


class NetworkDefinitionEdit(UpdateView):
    """
    Details for a given network definition.
    """
    template_name = 'editor/networkdefinition_form.html'
    model = NetworkDefinition
    form_class = NetworkDefinitionForm
    success_url = reverse_lazy('home')


class NetworkDefinitionCreate(CreateView):
    template_name = 'editor/networkdefinition_form.html'
    model = NetworkDefinition
    form_class = NetworkDefinitionForm
    success_url = reverse_lazy('home')


class NetworkDefinitionDelete(View):
    http_method_names = [u'delete']

    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        if request.method.lower() in self.http_method_names:
            handler = getattr(self, request.method.lower(), self.http_method_not_allowed)
        else:
            handler = self.http_method_not_allowed
        return handler(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        try:
            if request.method.lower() == 'delete':
                pk = self.kwargs.get('pk', None)
                if pk:
                    obj = get_object_or_404(NetworkDefinition, pk=pk)
                    obj.delete()
            return HttpResponse(204)
        except:
            logger.exception("Failed to delete definition!")


class NetworkParameterEdit(UpdateView):
    """
    Edits a given network definition's parameters.
    """
    template_name = 'editor/parameter_form.html'
    model = NetworkDefinition
    form_class = NetworkDefinitionParameterForm
    success_url = reverse_lazy('home')

    def get(self, request, *args, **kwargs):
        self.object = self.get_object()
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        parameters = ParameterFormSet(instance=self.object)
        return self.render_to_response(self.get_context_data(form=form, parameters=parameters))

    def post(self, request, *args, **kwargs):
        self.object = self.get_object()
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        parameters = ParameterFormSet(self.request.POST, instance=self.object)

        if form.is_valid() and parameters.is_valid():
            return self.form_valid(form, parameters)
        return self.form_invalid(form, parameters)

    def form_invalid(self, form, parameters):
        return self.render_to_response(
            self.get_context_data(form=form, parameters=parameters))

    def form_valid(self, form, parameters):
        instance = form.save(commit=True)
        parameters.instance = instance
        parameters.save(commit=True)
        return super(NetworkParameterEdit, self).form_valid(form)

    def get_context_data(self, **kwargs):
        context = super(NetworkParameterEdit, self).get_context_data(**kwargs)
        context['name'] = self.object.name
        return context
