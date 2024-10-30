# Generated by Django 4.2.16 on 2024-10-28 23:30

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0017_remove_cuestionario_nivel_pregunta_sub_tema_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='pregunta',
            name='sub_tema',
        ),
        migrations.AlterField(
            model_name='cuestionario',
            name='materia',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='cuestionario',
            name='usuario',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='cuestionarios', to=settings.AUTH_USER_MODEL),
        ),
    ]
