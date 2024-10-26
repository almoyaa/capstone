# Generated by Django 4.2.16 on 2024-10-25 21:01

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0016_conocimiento_remove_pregunta_nivel_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='cuestionario',
            name='nivel',
        ),
        migrations.AddField(
            model_name='pregunta',
            name='sub_tema',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AlterField(
            model_name='cuestionario',
            name='materia',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='materia', to='api.materia'),
        ),
        migrations.AlterField(
            model_name='cuestionario',
            name='usuario',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='usuario', to=settings.AUTH_USER_MODEL),
        ),
    ]