from django.contrib import admin
from .models import Usuario, Cuestionario, Pregunta, Respuesta, Resultado

class CuestionarioInline(admin.TabularInline):
    model = Cuestionario
    extra = 1  # Número de formularios adicionales vacíos para agregar más cuestionarios

class UsuarioAdmin(admin.ModelAdmin):
    list_display = ('email', 'nombre', 'tipo', 'estado', 'is_active', 'is_staff')
    inlines = [CuestionarioInline]  # Añade la vista inline para cuestionarios

class CuestionarioAdmin(admin.ModelAdmin):
    list_display = ('titulo', 'descripcion', 'nivel', 'materia', 'usuario')
    list_filter = ('usuario', 'nivel', 'materia')  # Filtros para la lista de cuestionarios

class PreguntaAdmin(admin.ModelAdmin):
    list_display = ('texto_pregunta', 'nivel', 'materia')
    list_filter = ('nivel', 'materia')

class RespuestaAdmin(admin.ModelAdmin):
    list_display = ('texto_respuesta', 'es_correcta', 'pregunta')
    list_filter = ('pregunta', 'es_correcta')

class ResultadoAdmin(admin.ModelAdmin):
    list_display = ('usuario', 'cuestionario', 'fecha', 'calificacion')
    list_filter = ('usuario', 'cuestionario')

class PreguntaInline(admin.TabularInline):
    model = Cuestionario.preguntas.through
    extra = 1


# Registra los modelos en el panel de administración
admin.site.register(Usuario, UsuarioAdmin)
admin.site.register(Cuestionario, CuestionarioAdmin)
admin.site.register(Pregunta, PreguntaAdmin)
admin.site.register(Respuesta, RespuestaAdmin)
admin.site.register(Resultado, ResultadoAdmin)
