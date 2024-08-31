from django.urls import path
from .views import UsuarioCreateView, CuestionarioCreateView

urlpatterns = [
    path('usuarios/', UsuarioCreateView.as_view(), name='crear_usuario'),
    path('cuestionarios/', CuestionarioCreateView.as_view(), name='cuestionario-create'),
]
