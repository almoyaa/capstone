from django.contrib import admin
from .models import Usuario, Cuestionario, Pregunta, Respuesta, Resultado, Materia, Tema, Conocimiento

class CuestionarioInline(admin.TabularInline):
    model = Cuestionario
    extra = 1  # Número de formularios adicionales vacíos para agregar más cuestionarios

class UsuarioAdmin(admin.ModelAdmin):
    list_display = ('email', 'nombre', 'tipo', 'estado', 'is_active', 'is_staff')
    inlines = [CuestionarioInline]  # Añade la vista inline para cuestionarios

class RespuestaFiltro(admin.TabularInline):
    model = Cuestionario.preguntas.through  # Usa la relación ManyToMany intermedia
    extra = 0
    verbose_name = 'Pregunta'
    verbose_name_plural = 'Preguntas'

class CuestionarioAdmin(admin.ModelAdmin):

    inlines = [RespuestaFiltro]  # Incluye el Inline dentro de Cuestionario
    readonly_fields = ('respuestas_correctas',)
    list_display = ('id','titulo', 'descripcion','materia', 'usuario')
    list_filter = ('usuario', 'materia')  # Filtros para la lista de cuestionarios

class PreguntaAdmin(admin.ModelAdmin):
    list_display = ('texto_pregunta','materia')

class MateriaAdmin(admin.ModelAdmin):

    list_display = ('id','nombre')
class TemaAdmin(admin.ModelAdmin):
    list_display = ('id','nombre','mostrar_conocimientos')
    list_filter = ['nombre']

class ConocimientoAdmin(admin.ModelAdmin):
    list_display = ('nombre','id')
    

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
admin.site.register(Materia, MateriaAdmin)
admin.site.register(Tema, TemaAdmin)
admin.site.register(Conocimiento, ConocimientoAdmin)