from django.db import models
import os

class UserDetails(models.Model):
    name = models.CharField(null=True,max_length=100)
    email = models.CharField(null = True,max_length=20)
    password = models.CharField(null=True,max_length=100)