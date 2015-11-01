# -*-*- encoding: utf-8 -*-*-
from __future__ import absolute_import, unicode_literals
from django import forms
from django.forms.models import inlineformset_factory
from .models import NetworkDefinition, NetworkParameter


class NetworkDefinitionForm(forms.ModelForm):

    def __init__(self, *args, **kwargs):
        super(NetworkDefinitionForm, self).__init__(*args, **kwargs)

        # Definition field should be hidden
        self.fields['definition'].widget = forms.HiddenInput()

    class Meta:
        model = NetworkDefinition
        fields = ('name', 'definition',)


class NetworkDefinitionParameterForm(forms.ModelForm):

    class Meta:
        model = NetworkDefinition
        fields = ()


class NetworkParameterForm(forms.ModelForm):

    def __init__(self, *args, **kwargs):
        super(NetworkParameterForm, self).__init__(*args, **kwargs)
        self.fields['ptype'].widget.attrs['style'] = 'display:none;'
        self.fields['value'].widget.attrs['step'] = 0.1

    class Meta:
        model = NetworkParameter
        fields = ('ptype', 'value',)
        labels = {
            'ptype': 'Type',
        }

ParameterFormSet = inlineformset_factory(
    NetworkDefinition, NetworkParameter,
    form=NetworkParameterForm, extra=0,
    can_order=False, can_delete=False)
