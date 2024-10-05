from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager

class UserAccountManager(BaseUserManager):

    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('El usuario debe tener un correo electrónico')
        
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save()
        return user

    def create_superuser(self, email, password, **extra_fields):
        user = self.create_user(email, password, **extra_fields)
        user.is_superuser = True
        user.is_staff = True
        user.save()
        return user

class Usuario(AbstractBaseUser, PermissionsMixin):
    TIPO_CHOICES = (
        ('admin', 'Admin'),
        ('invitado', 'Invitado'),
        ('estudiante', 'Estudiante'),
        ('tutor', 'Tutor'),
    )
    id = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length=100, blank=True, null=True)
    email = models.EmailField(max_length=255, unique=True)
    tipo = models.CharField(max_length=100, blank=True, null=True, choices=TIPO_CHOICES, default='estudiante')
    estado = models.BooleanField(default=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    objects = UserAccountManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []
    
    def __str__(self):
        return self.email

class Pregunta(models.Model):
    NIVEL_CHOICES = [
        ('1ero Medio', '1dro Medio'),
        ('2do Medio', '2do Medio'),
        ('3ero Medio', '3ero Medio'),
        ('4to Medio', '4to Medio'),
    ]

    MATERIA_CHOICES = [
        ('Biologia', 'Biologia'),
        ('Fisica', 'Fisica'),
        ('Quimica', 'Quimica'),
        ('Matematica','Matematica'),
        ('Lenguaje','Lenguaje')
    ]

    id = models.AutoField(primary_key=True)
    texto_pregunta = models.TextField()
    nivel = models.CharField(max_length=10, choices=NIVEL_CHOICES)
    materia = models.CharField(max_length=10, choices=MATERIA_CHOICES)

    def __str__(self):
        return self.texto_pregunta

class Cuestionario(models.Model):
    NIVEL_CHOICES = [
        ('1ero Medio', '1dro Medio'),
        ('2do Medio', '2do Medio'),
        ('3ero Medio', '3ero Medio'),
        ('4to Medio', '4to Medio'),
    ]

    MATERIA_CHOICES = [
        ('Biologia', 'Biologia'),
        ('Fisica', 'Fisica'),
        ('Quimica', 'Quimica'),
        ('Matematica','Matematica'),
        ('Lenguaje','Lenguaje')
    ]

    id = models.AutoField(primary_key=True)
    titulo = models.CharField(max_length=100)
    descripcion = models.TextField()
    nivel = models.CharField(max_length=10, choices=NIVEL_CHOICES)
    materia = models.CharField(max_length=10, choices=MATERIA_CHOICES)
    usuario = models.ForeignKey(Usuario,null=True, blank=True, on_delete=models.CASCADE, related_name='cuestionarios')
    preguntas = models.ManyToManyField(Pregunta, blank=True)

    def __str__(self):
        return self.titulo

class Respuesta(models.Model):
    id = models.AutoField(primary_key=True)
    texto_respuesta = models.TextField()
    es_correcta = models.BooleanField(null=True)
    pregunta = models.ForeignKey(Pregunta, related_name='respuestas', on_delete=models.CASCADE)

    def __str__(self):
        return self.texto_respuesta

class Resultado(models.Model):
    id = models.AutoField(primary_key=True)
    usuario = models.ForeignKey(Usuario, on_delete=models.CASCADE)
    cuestionario = models.ForeignKey(Cuestionario, on_delete=models.CASCADE)
    fecha = models.DateTimeField(auto_now_add=True)
    calificacion = models.FloatField()

    def __str__(self):
        return f'Examen de {self.usuario.email} - {self.cuestionario.titulo}'
