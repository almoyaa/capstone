# Generated by Django 4.2.16 on 2024-11-19 20:07

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.CreateModel(
            name='Usuario',
            fields=[
                ('password', models.CharField(max_length=128, verbose_name='password')),
                ('last_login', models.DateTimeField(blank=True, null=True, verbose_name='last login')),
                ('is_superuser', models.BooleanField(default=False, help_text='Designates that this user has all permissions without explicitly assigning them.', verbose_name='superuser status')),
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(blank=True, max_length=100, null=True)),
                ('email', models.EmailField(max_length=255, unique=True)),
                ('tipo', models.CharField(blank=True, choices=[('admin', 'Admin'), ('invitado', 'Invitado'), ('estudiante', 'Estudiante'), ('tutor', 'Tutor')], default='estudiante', max_length=100, null=True)),
                ('estado', models.BooleanField(default=True)),
                ('is_active', models.BooleanField(default=True)),
                ('is_staff', models.BooleanField(default=False)),
                ('groups', models.ManyToManyField(blank=True, help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.', related_name='user_set', related_query_name='user', to='auth.group', verbose_name='groups')),
                ('user_permissions', models.ManyToManyField(blank=True, help_text='Specific permissions for this user.', related_name='user_set', related_query_name='user', to='auth.permission', verbose_name='user permissions')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Conocimiento',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(blank=True, max_length=255, null=True)),
                ('descripcion', models.TextField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Cuestionario',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('titulo', models.CharField(max_length=100)),
                ('descripcion', models.TextField()),
                ('materia', models.CharField(blank=True, max_length=100, null=True)),
                ('preguntas_correctas', models.IntegerField(blank=True, null=True)),
                ('fecha_creacion', models.DateTimeField(auto_now_add=True)),
                ('fecha_realizacion', models.DateTimeField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Materia',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(blank=True, max_length=100, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Pregunta',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('texto_pregunta', models.TextField()),
                ('materia', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='Materia', to='api.materia')),
            ],
        ),
        migrations.CreateModel(
            name='Tema',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(blank=True, max_length=255, null=True)),
                ('conocimiento', models.ManyToManyField(to='api.conocimiento', verbose_name='Conocimientos')),
            ],
        ),
        migrations.CreateModel(
            name='Resultado',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('fecha', models.DateTimeField(auto_now_add=True)),
                ('calificacion', models.FloatField()),
                ('cuestionario', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.cuestionario')),
                ('usuario', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Respuesta',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('texto_respuesta', models.TextField()),
                ('es_correcta', models.BooleanField(null=True)),
                ('pregunta', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='respuestas', to='api.pregunta')),
            ],
        ),
        migrations.AddField(
            model_name='pregunta',
            name='tema',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='api.tema', verbose_name='Tema'),
        ),
        migrations.AddField(
            model_name='materia',
            name='tema',
            field=models.ManyToManyField(to='api.tema', verbose_name='Temas'),
        ),
        migrations.AddField(
            model_name='cuestionario',
            name='preguntas',
            field=models.ManyToManyField(blank=True, to='api.pregunta'),
        ),
        migrations.AddField(
            model_name='cuestionario',
            name='usuario',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='cuestionarios', to=settings.AUTH_USER_MODEL),
        ),
    ]
