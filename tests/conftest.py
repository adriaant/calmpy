# -*-*- encoding: utf-8 -*-*-
# pylint: disable=E1101
from __future__ import unicode_literals
import pytest
from django.core.management import call_command


@pytest.fixture
def network():
    call_command('loaddata', 'tests/fixtures/network.json')
