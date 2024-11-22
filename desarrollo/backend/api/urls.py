from django.urls import path
from .views import UsuarioCreateView, CuestionarioCreateView, UsuarioAllView, CuestionarioAllView, PreguntaCreateView, PreguntaAllView, crear_preguntas, MateriaListView, CuestionarioListView, historial_usuario, HistorialTemplateView, retro, comentario_cuestionario, obtener_progreso, get_chart

urlpatterns = [
    path('crear-usuario/', UsuarioCreateView.as_view(), name='crear-usuario'),
    path('usuarios/', UsuarioAllView.as_view(), name='ver-usuarios'),
    path('crear-cuestionario/', CuestionarioCreateView.as_view(), name='crear-cuestionario-api'),
    path('cuestionarios/', CuestionarioAllView.as_view(), name='ver-cuestionario'),
    path('crear-pregunta/', PreguntaCreateView.as_view(),name='crear-pregunta'),
    path('preguntas/', PreguntaAllView.as_view(),name='ver-pregunta'),
    path('materias/',MateriaListView.as_view(), name='ver-materias'),
    path('crear-preguntas/', crear_preguntas,name='crear-preguntas'),
    path('cuestionarios/<int:usuario_id>/', CuestionarioListView.as_view(), name='cuestionarios-usuario'),
    path('historial/', historial_usuario, name='historial-datos'),
    path('retro/',retro, name='crear-retro'),
    path('comentario_cuestionario',comentario_cuestionario,name='comentario_cuestionario'),
    path('progreso/', obtener_progreso, name='obtener-progreso'),
    path('get_chart/', get_chart, name='get_chart')
]