from django.contrib import admin
from django.urls import path, include
from api.views import IndexTemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),  # Incluye las URLs de la aplicación `api`
    path('', IndexTemplateView.as_view(),name='index-page'),
]
