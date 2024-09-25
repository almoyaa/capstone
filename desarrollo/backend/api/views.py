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

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@csrf_exempt
def chatgpt_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            materia = data.get('materia')
            nivel = data.get('nivel')
            prompt = f"""
            Genera una pregunta de opción múltiple sobre {materia} para un estudiante de {nivel} bajo el temario de estudio de la PAES CHILE 2023 o la más actualizada que tengas. La pregunta debe tener 5 opciones de respuesta y una es la correcta aleatoriamente. Devuélveme el resultado en el siguiente formato JSON:
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

            pregunta = Pregunta.objects.create(
                texto_pregunta=question_data["pregunta"],
                nivel=nivel,
                materia=materia
            )

            #SE GUARDA RESPUESTA EN BASE DE DATOS
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



class IndexTemplateView(TemplateView):
    template_name = "index.html"

class PreguntaTemplateView(TemplateView):
    template_name="pregunta.html"