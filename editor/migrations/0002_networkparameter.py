# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import enumfields.fields
import editor.models


class Migration(migrations.Migration):

    dependencies = [
        ('editor', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='NetworkParameter',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('ptype', enumfields.fields.EnumField(max_length=10, enum=editor.models.ParameterType)),
                ('value', models.FloatField()),
                ('network', models.ForeignKey(related_name='parameters', to='editor.NetworkDefinition')),
            ],
        ),
    ]
