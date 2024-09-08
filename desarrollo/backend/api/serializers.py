from rest_framework import serializers
from .models import Usuario, Cuestionario, Pregunta, Respuesta

class UsuarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = Usuario
        fields = ['email', 'password', 'nombre', 'tipo', 'estado']
        extra_kwargs = {
            'password': {'write_only': True},
        }

    def create(self, validated_data):
        user = Usuario.objects.create_user(
            email=validated_data['email'],
            password=validated_data['password'],
            nombre=validated_data.get('nombre'),
            tipo=validated_data.get('tipo'),
            estado=validated_data.get('estado')
        )
        return user

class RespuestaSerializer(serializers.ModelSerializer):

    class Meta:
        model = Respuesta
        fields = ['id','texto_respuesta','es_correcta',]

class PreguntaSerializer(serializers.ModelSerializer):
    respuestas = RespuestaSerializer(many=True, read_only=True)
    class Meta:
        model = Pregunta
        fields = ['id', 'texto_pregunta', 'nivel', 'materia','respuestas']

class CuestionarioSerializer(serializers.ModelSerializer):
    preguntas = serializers.PrimaryKeyRelatedField(queryset=Pregunta.objects.all(), many=True, required=False)

    class Meta:
        model = Cuestionario
        fields = ['id', 'titulo', 'descripcion', 'nivel', 'materia', 'preguntas']

    def create(self, validated_data):
        preguntas_data = validated_data.pop('preguntas', [])
        cuestionario = Cuestionario.objects.create(**validated_data)
        cuestionario.preguntas.set(preguntas_data)  # Añade las preguntas al cuestionario
        return cuestionario