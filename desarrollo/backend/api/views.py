import os
import json
import openai
from django.shortcuts import render
from rest_framework import generics
from .models import Usuario, Cuestionario, Pregunta, Respuesta
from .serializers import UsuarioSerializer, CuestionarioSerializer, PreguntaSerializer
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.http import JsonResponse
from dotenv import load_dotenv
from django.views.generic import TemplateView
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
class PreguntaDetailView(generics.RetrieveAPIView):
    queryset = Pregunta.objects.all()
    serializer_class = PreguntaSerializer

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@csrf_exempt
def chatgpt_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            materia = data.get('materia')
            nivel = data.get('nivel')

            nivel_mapping = {
                "1ero Medio": "1ero Medio",
                "2do Medio": "2do Medio",
                "3ero Medio": "3ero Medio",
                "4to Medio": "4to Medio"
            }

            materia_mapping = {
                "Biologia": "Biologia",
                "Fisica": "Fisica",
                "Quimica": "Quimica",
                "Lenguaje":"Lenguaje",
                "Matematica":"Matematica"
            }

            # Convertir los valores de materia y nivel a las claves correctas
            nivel_clave = nivel_mapping.get(nivel)
            materia_clave = materia_mapping.get(materia)

            if not nivel_clave or not materia_clave:
                return JsonResponse({"error": "Materia o nivel inválidos."}, status=400)

            # Prompt para generar la pregunta usando OpenAI
            prompt = f"""
            Genera una pregunta de opción múltiple sobre {materia} para un estudiante correspondiende a la materia de estudio de {nivel} de la PAES CHILE 2023 o la más actualizada que tengas. La pregunta debe tener 5 opciones de respuesta, de dificultad alta, y una es la correcta aleatoriamente. Devuélveme el resultado en el siguiente formato JSON:
            {{
                "pregunta": "Texto de la pregunta",
                "opciones": [
                    {{"texto": "Opción 1", "es_correcta": none}},
                    {{"texto": "Opción 2", "es_correcta": none}},
                    {{"texto": "Opción 3", "es_correcta": none}},
                    {{"texto": "Opción 4", "es_correcta": none}},
                    {{"texto": "Opción 5", "es_correcta": none}}
                ]
            }}
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )

            content = response.choices[0].message.content.strip()
            question_data = json.loads(content)
            print(question_data)

            # Crear la pregunta en la base de datos
            pregunta = Pregunta.objects.create(
                texto_pregunta=question_data["pregunta"],
                nivel=nivel_clave,
                materia=materia_clave
            )

            # Crear y asociar las respuestas con la pregunta
            for opcion in question_data["opciones"]:
                Respuesta.objects.create(
                    texto_respuesta=opcion["texto"],
                    es_correcta=opcion["es_correcta"],
                    pregunta=pregunta
                )

            # Preparar el JSON de respuesta
            respuestas = [
                {
                    "id": respuesta.id,
                    "texto_respuesta": respuesta.texto_respuesta,
                    "es_correcta": respuesta.es_correcta
                }
                for respuesta in pregunta.respuestas.all()  # Utilizamos el related_name
            ]

            pregunta_json = {
                "id": pregunta.id,
                "texto_pregunta": pregunta.texto_pregunta,
                "nivel": pregunta.nivel,
                "materia": pregunta.materia,
                "respuestas": respuestas
            }

            return JsonResponse(pregunta_json, status=201)

        except json.JSONDecodeError as e:
            return JsonResponse({"error": "Error al decodificar JSON: " + str(e)}, status=500)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Método de solicitud no permitido."}, status=405)


class IndexTemplateView(TemplateView):
    template_name = "index.html"

class PreguntaTemplateView(TemplateView):
    template_name="pregunta.html"