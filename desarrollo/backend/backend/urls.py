from django.contrib import admin
from django.urls import path, include
from api.views import IndexTemplateView, PreguntaTemplateView, CrearCuestionarioView, HistorialTemplateView, RetroalimentacionTemplateView, historial_usuario

#URL DEL front
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),  # Incluye las URLs de la aplicación `api`
    path('', IndexTemplateView.as_view(),name='index-page'),
    path('pregunta/', PreguntaTemplateView.as_view(), name='mostrar-pregunta'),
    path('crear-cuestionario/', CrearCuestionarioView.as_view(), name='crear-cuestionario'),
    path('historial/', historial_usuario, name='historial'),  # Añadir aquí la ruta de historial
    path('retroalimentacion/',RetroalimentacionTemplateView.as_view(), name='mostrar-retro')
]
