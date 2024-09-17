import os
import json
import openai
from django.shortcuts import render
from rest_framework import generics
from .models import Usuario, Cuestionario, Pregunta
from .serializers import UsuarioSerializer, CuestionarioSerializer, PreguntaSerializer
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.http import JsonResponse
from dotenv import load_dotenv


load_dotenv()

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


client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@csrf_exempt
def chatgpt_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            materia = data.get('materia', 'Biología')  # Puedes ajustar este valor
            nivel = data.get('nivel', '1M')  # Puedes ajustar este valor

            prompt = f"""
            Genera una pregunta de opción múltiple sobre {materia} para estudiantes de {nivel}. La pregunta debe tener 5 opciones de respuesta. Devuélveme el resultado en el siguiente formato JSON:

            {{
                "pregunta": "Texto de la pregunta",
                "opciones": [
                    {{"texto": "Opción 1", "es_correcta": false}},
                    {{"texto": "Opción 2", "es_correcta": false}},
                    {{"texto": "Opción 3", "es_correcta": true}},
                    {{"texto": "Opción 4", "es_correcta": false}},
                    {{"texto": "Opción 5", "es_correcta": false}}
                ]
            }}
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )

            content = response.choices[0].message.content.strip()

            # Convertir la respuesta en JSON
            question_data = json.loads(content)

            # Guardar la pregunta en la base de datos
            pregunta = Pregunta.objects.create(
                texto_pregunta=question_data["pregunta"],
                nivel=nivel,
                materia=materia
            )

            # Guardar las respuestas en la base de datos
            for opcion in question_data["opciones"]:
                Respuesta.objects.create(
                    texto_respuesta=opcion["texto"],
                    es_correcta=opcion["es_correcta"],
                    pregunta=pregunta
                )

            return JsonResponse({"mensaje": "Pregunta creada con éxito"})
        
        except json.JSONDecodeError as e:
            return JsonResponse({"error": "Error al decodificar JSON: " + str(e)}, status=500)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Método de solicitud no permitido."}, status=405)
