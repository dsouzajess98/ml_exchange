# Generated by Django 3.0.6 on 2020-05-18 11:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_exchange', '0003_remove_smartcontract_account_no'),
    ]

    operations = [
        migrations.AlterField(
            model_name='smartcontract',
            name='submission_phase',
            field=models.BooleanField(default=False),
        ),
    ]
