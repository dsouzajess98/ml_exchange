# Generated by Django 3.0.6 on 2020-06-19 17:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_exchange', '0015_submission_created_at'),
    ]

    operations = [
        migrations.AddField(
            model_name='smartcontract',
            name='pseudo_user',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
        migrations.AddField(
            model_name='submission',
            name='pseudo_user',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
    ]
