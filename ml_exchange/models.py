from django.conf import settings
from django.db import models
from django.utils import timezone
from ipfs_storage import InterPlanetaryFileSystemStorage


class SmartContract(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="creator")
    name = models.CharField(max_length=256)
    created_date = models.DateTimeField(default=timezone.now)
    submission_period_days = models.IntegerField(null=True, blank=True)
    submission_period_hours = models.IntegerField(null=True, blank=True)
    submission_period_minutes = models.IntegerField(null=True, blank=True)
    submission_phase = models.BooleanField(default=False)
    testing_phase = models.BooleanField(default=False)
    contract_active = models.BooleanField(default=True)
    best_submission = models.FloatField(null=True, blank=True)
    best_submission_user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True, related_name="best_submission")
    document = models.FileField(null=True, blank=True)
    dataset = models.FileField(null=True, blank=True)

    def __str__(self):
        return self.name


class run_init(models.Model):
    done = models.BooleanField(default=False)


class Submission(models.Model):
    contract = models.ForeignKey(SmartContract, on_delete=models.CASCADE)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    training_data = models.FileField(blank=True, null=True)
    testing_data = models.FileField(blank=True, null=True)
    accuracy = models.FloatField(blank=True, null=True)
    viewed_training_data = models.BooleanField(default=False)
    model = models.FileField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)


class Request(models.Model):
    # 0 - Same requirements 1 - Dataset help 2 - Reopen contract/dataset help 3 - Download previous model
    from_user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="from_user")
    to_user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="to_user")
    request_type = models.IntegerField(null=True, blank=True)
    approved = models.BooleanField(default=False)
    dataset = models.FileField(blank=True, null=True)
    document = models.FileField(blank=True, null=True)
    contract = models.ForeignKey(SmartContract, on_delete=models.CASCADE,null=True)
    viewed = models.BooleanField(default=False)
    ether = models.FloatField(null=True, blank=True)


class pseudo(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    pseudo_user = models.CharField(max_length=200, null=True, blank=True)
