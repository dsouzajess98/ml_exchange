# Generated by Django 3.0.6 on 2020-05-17 07:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_exchange', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='smartcontract',
            name='submission_phase',
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name='smartcontract',
            name='testing_phase',
            field=models.BooleanField(default=False),
        ),
    ]