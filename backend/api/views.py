from django.shortcuts import render
from rest_framework import generics
from .models import Usuario, Cuestionario
from .serializers import UsuarioSerializer, CuestionarioSerializer

class UsuarioCreateView(generics.CreateAPIView):
    queryset = Usuario.objects.all()
    serializer_class = UsuarioSerializer

class CuestionarioCreateView(generics.CreateAPIView):
    queryset = Cuestionario.objects.all()
    serializer_class = CuestionarioSerializer