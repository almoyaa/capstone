from django.shortcuts import render
from rest_framework import generics
from .models import Usuario, Cuestionario, Pregunta
from .serializers import UsuarioSerializer, CuestionarioSerializer, PreguntaSerializer

class UsuarioCreateView(generics.CreateAPIView):
    queryset = Usuario.objects.all()
    serializer_class = UsuarioSerializer
class UsuarioAllView(generics.ListAPIView):
    queryset = Usuario.objects.all()
    serializer_class = UsuarioSerializer

class CuestionarioCreateView(generics.CreateAPIView):
    queryset = Cuestionario.objects.all()
    serializer_class = CuestionarioSerializer

class CuestionarioAllView(generics.CreateAPIView):
    queryset = Cuestionario.objects.all()
    serializer_class = CuestionarioSerializer

class PreguntaCreateView(generics.CreateAPIView):
    queryset = Pregunta.objects.all()
    serializer_class = PreguntaSerializer

class PreguntaAllView(generics.ListAPIView):
    queryset = Pregunta.objects.all()
    serializer_class = PreguntaSerializer
