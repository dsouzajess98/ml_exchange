# Generated by Django 3.0.6 on 2020-05-17 07:27

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ml_exchange', '0002_auto_20200517_0722'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='smartcontract',
            name='account_no',
        ),
    ]