from django.urls import path
from .views import UsuarioCreateView, CuestionarioCreateView, UsuarioAllView, CuestionarioAllView, PreguntaCreateView, PreguntaAllView

urlpatterns = [
    path('crear-usuario/', UsuarioCreateView.as_view(), name='crear-usuario'),
    path('usuarios/', UsuarioAllView.as_view(), name='ver-usuarios'),
    path('crear-cuestionario/', CuestionarioCreateView.as_view(), name='crear-cuestionario'),
    path('cuestionarios/', CuestionarioAllView.as_view(), name='ver-cuestionario'),
    path('crear-pregunta/', PreguntaCreateView.as_view(),name='crear-pregunta'),
    path('preguntas/', PreguntaAllView.as_view(),name='ver-pregunta')
]