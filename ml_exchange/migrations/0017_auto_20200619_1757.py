# Generated by Django 3.0.6 on 2020-06-19 17:57

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('ml_exchange', '0016_auto_20200619_1751'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='smartcontract',
            name='pseudo_user',
        ),
        migrations.RemoveField(
            model_name='submission',
            name='pseudo_user',
        ),
        migrations.RemoveField(
            model_name='submission',
            name='user',
        ),
        migrations.CreateModel(
            name='pseudo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pseudo_user', models.CharField(blank=True, max_length=200, null=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
