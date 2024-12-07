import os
import json
import openai
import time
import traceback
from datetime import datetime
from collections import defaultdict, Counter
from random import randrange
from dotenv import load_dotenv

from django.shortcuts import render, get_object_or_404, redirect
from rest_framework import generics, status
from rest_framework.response import Response
from django.utils.decorators import method_decorator
from django.utils.safestring import mark_safe
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView
from django.conf import settings
from django.http import JsonResponse
from django.db.models import Count

from .models import Usuario, Cuestionario, Pregunta, Respuesta, Materia, Tema, Conocimiento
from .serializers import UsuarioSerializer, CuestionarioSerializer, PreguntaSerializer, MateriaSerializer


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.schema import Document  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent

from sentence_transformers import SentenceTransformer, util

import matplotlib.pyplot as plt
from io import BytesIO
import base64

load_dotenv()
def cargar_pdfs_desde_carpeta(carpeta, embeddings_model):
    todos_los_documentos = []

    # Crear una instancia de RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Tamaño del fragmento
        chunk_overlap=200  # Superposición entre fragmentos
    )

    # Recorrer todos los archivos de la carpeta
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".pdf"):
            ruta_completa = os.path.join(carpeta, archivo)
            print(f"Cargando {ruta_completa}")

            # Cargar y dividir el PDF en páginas
            loader = PyPDFLoader(ruta_completa)
            paginas = loader.load_and_split()

            # Procesar cada página
            for pagina in paginas:
                # Verificar que 'pagina' sea un objeto Document
                if hasattr(pagina, 'page_content'):
                    # Aplicar el text_splitter a cada página
                    fragments = text_splitter.split_text(pagina.page_content)
                    # Crear y agregar objetos Document para cada fragmento
                    for fragment in fragments:
                        doc = Document(page_content=fragment, metadata={"source": ruta_completa})
                        todos_los_documentos.append(doc)
                else:
                    print(f"Advertencia: La página no tiene el atributo 'page_content': {pagina}")

    # Crear el índice vectorial FAISS con todos los documentos combinados
    faiss_index = FAISS.from_documents(todos_los_documentos, embeddings_model)

    # Retornar el retriever
    return faiss_index.as_retriever()

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")
<<<<<<< HEAD
CARPETAS_PDF = "C:/Users/Alejandro/capstone/desarrollo/backend/api/pdf"
=======
CARPETAS_PDF = "/Users/sebastian/Downloads/Contenidos"
>>>>>>> 8143b3b035f6260ebffc42632ea57935d91f368e
RETRIEVER = cargar_pdfs_desde_carpeta(CARPETAS_PDF, EMBEDDINGS)
RETRIEVER_TOOL = create_retriever_tool(
                RETRIEVER,
                "Temarios_prueba_PAES",
                "Recurso para establecer el contenido con los que se van evaluar los conocimientos del alumno, para cada temario.",
            )
TOOLS = [RETRIEVER_TOOL]

model_embedding = SentenceTransformer('all-MiniLM-L6-v2')

class UsuarioCreateView(generics.CreateAPIView):
    queryset = Usuario.objects.all()
    serializer_class = UsuarioSerializer
class UsuarioAllView(generics.ListAPIView):
    queryset = Usuario.objects.all()
    serializer_class = UsuarioSerializer

class CuestionarioCreateView(generics.CreateAPIView):
    queryset = Cuestionario.objects.all()
    serializer_class = CuestionarioSerializer

    def create(self, request, *args, **kwargs):
        # Obtener la fecha y hora actual en formato 'DD/MM/YYYY HH:MM'
        fecha_actual = datetime.now().strftime("%d/%m/%Y %H:%M")
        
        # Crear un nuevo título con la fecha y hora
        titulo_con_fecha = f"Cuestionario de {request.data.get('materia')} - {fecha_actual}"

        # Agregar el nuevo título al data del request antes de pasar al serializer
        request.data['titulo'] = titulo_con_fecha
        request.data['fecha_creacion'] = datetime.now()

        # Pasar los datos al serializer
        serializer = self.get_serializer(data=request.data)
        
        if serializer.is_valid():
            cuestionario = serializer.save()  # Guardar el cuestionario
            response_data = {
                'success': True,
                'message': 'Cuestionario guardado exitosamente.',
                'data': serializer.data  # Esto incluye el serializador del nuevo objeto
            }
            return Response(response_data, status=status.HTTP_201_CREATED)
        else:
            error_response = {
                'success': False,
                'errors': serializer.errors
            }
            return Response(error_response, status=status.HTTP_400_BAD_REQUEST)

class CuestionarioAllView(generics.ListCreateAPIView):
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
    
class MateriaListView(generics.ListAPIView):
    queryset = Materia.objects.all()
    serializer_class = MateriaSerializer

class CuestionarioListView(generics.ListAPIView):
    serializer_class = CuestionarioSerializer

    def get_queryset(self):
        # Obtener el usuario a través de un parámetro en la URL
        usuario_id = self.kwargs.get('usuario_id')  # Asegúrate de que 'usuario_id' sea el parámetro en tu URL
        return Cuestionario.objects.filter(usuario_id=usuario_id)



client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



class IndexTemplateView(TemplateView):
    template_name = "index.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['materias'] = Materia.objects.all()
        return context

class PreguntaTemplateView(TemplateView):
    template_name = "pregunta.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Cargar preguntas de la sesión
        preguntas = self.request.session.get('preguntas_guardadas', [])
        context['preguntas_json'] = json.dumps(preguntas)  # Convertir a JSON
        return context

        


class HistorialTemplateView(TemplateView):
    template_name = "historial.html"

class CrearCuestionarioView(TemplateView):
    template_name="crear-cuestionario.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        materias = Materia.objects.all()
        context['materias'] = materias
        return context

class RetroalimentacionTemplateView(TemplateView):
    template_name = "retroalimentacion.html"

    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)

    def get_respuestas_usuario(self, preguntas, respuestas):
        """Procesa las respuestas y devuelve una lista detallada y un conteo de errores por tema."""
        respuestas_usuario = []
        respuestas_erroneas_por_tema = defaultdict(int)

        for pregunta in preguntas:
            respuesta_usuario = respuestas.filter(pregunta=pregunta).first()
            respuesta_correcta = pregunta.respuestas.filter(es_correcta=True).first()
            es_correcta = respuesta_usuario.es_correcta if respuesta_usuario else False

            if not es_correcta:
                tema = pregunta.tema.nombre
                respuestas_erroneas_por_tema[tema] += 1

            respuestas_usuario.append({
                "pregunta": pregunta,
                "respuesta_usuario": respuesta_usuario.texto_respuesta if respuesta_usuario else "No respondida",
                "es_correcta": es_correcta,
                "respuesta_correcta": respuesta_correcta.texto_respuesta if respuesta_correcta else "No especificada"
            })
        
        return respuestas_usuario, respuestas_erroneas_por_tema

    def post(self, request, *args, **kwargs):
        cuestionario_id = request.POST.get('cuestionario_id')
        cuestionario = get_object_or_404(Cuestionario, id=cuestionario_id)
        preguntas = cuestionario.preguntas.all()
        respuestas = cuestionario.respuestas_usuario.all()
        respuestas_usuario, respuestas_erroneas_por_tema = self.get_respuestas_usuario(preguntas, respuestas)

        respuestas_correctas = sum(1 for respuesta in respuestas if respuesta.es_correcta)
        respuestas_incorrectas = len(respuestas) - respuestas_correctas



        # CONVIERTO DICT A JSON PARA TRABAJAR EN JAVASCRIPT (GRAFICO)
        errores_por_tema_json = json.dumps(respuestas_erroneas_por_tema)
        context = self.get_context_data(
            cuestionario=cuestionario,
            preguntas=preguntas,
            respuestas_usuario=respuestas_usuario,
            errores_por_tema=errores_por_tema_json,
            errores_por_tema_tabla=dict(respuestas_erroneas_por_tema),
            grafico_data={
                "labels": ["Correctas", "Incorrectas"],
                "data": [respuestas_correctas, respuestas_incorrectas],
            },
        )

        return self.render_to_response(context)


class RetroPreguntaTemplateView(TemplateView):
    template_name = 'retro-pregunta.html'
    
    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def post(self, request, *args, **kwargs):
            # Obtén el ID de la pregunta y la respuesta del usuario desde el POST
            pregunta_id = request.POST.get('pregunta_id')
            respuesta_usuario = request.POST.get('respuesta_usuario')

            # Obtener la pregunta y la respuesta correcta
            pregunta = get_object_or_404(Pregunta, id=pregunta_id)
            materia = pregunta.materia
            respuesta_correcta = pregunta.respuestas.get(es_correcta=True).texto_respuesta

            prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """Eres un profesor de enseñanza media preparando alumnos para la evaluación PAES de Chile. Un alumno te entrega una pregunta con la respuesta correcta, y lo que respondio el alumno.
                            Indicale al alumno porque se equivoca, y formula 3 preguntas de dos alternativas, orientadas a la pregunta del estudiante.
                            [
                    {{
                        {{"comentario":"texto_comentario",
                        "preguntas":[
                        "pregunta": "Texto de la pregunta",
                        "tema":"tema correspondiente",
                        "opciones": [
                            {{"texto": "Opción 1", "es_correcta": null}},
                            {{"texto": "Opción 2", "es_correcta": null}}
                        ],
                        "pregunta": "Texto de la pregunta",
                        "tema":"tema correspondiente",
                        "opciones": [
                            {{"texto": "Opción 1", "es_correcta": null}},
                            {{"texto": "Opción 2", "es_correcta": null}}
                        ],
                        "pregunta": "Texto de la pregunta",
                        "tema":"tema correspondiente",
                        "opciones": [
                            {{"texto": "Opción 1", "es_correcta": null}},
                            {{"texto": "Opción 2", "es_correcta": null}}]
                        
                        }}
                        ]
                    }},
                    ...
                ]"""
                        ),
                        (
                "human",
                f"Pregunta:{pregunta}, Respuesta correcta: {respuesta_correcta}, Respuesta alumno: {respuesta_usuario}. ENTREGA SOLO' EN FORMATO JSON"
            ),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ]
                )

            seed = int(time.time())
            model = ChatOpenAI(temperature=0.3, model="gpt-4o", seed=seed)
            agent = create_openai_tools_agent(model, TOOLS, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)
            response = agent_executor.invoke({})
            output_text = response.get("output", "")
            clean_output = output_text.replace('```json\n', '').replace('```', '')
            retroalimentacionData = json.loads(clean_output)

            

            # Añadir al contexto para el template
            context = {
                'pregunta': pregunta,
                'respuesta_usuario': respuesta_usuario,
                'respuesta_correcta': respuesta_correcta,
                'retroalimentacion':retroalimentacionData
            }
            return self.render_to_response(context)

def similarity_sentence_transformers(text1, text2):
    """
    Calcula la similitud entre dos textos utilizando Sentence Transformers.
    """
    embedding1 = model_embedding.encode(text1, convert_to_tensor=True)
    embedding2 = model_embedding.encode(text2, convert_to_tensor=True)
    return util.cos_sim(embedding1, embedding2).item()



@csrf_exempt
def crear_preguntas(request):
    if request.method == 'POST':
        try:
            # Extraer los parámetros de la solicitud
            cantidad = int(request.POST.get("cantidad"))
            materia_nombre = request.POST.get("materia")
            materia = Materia.objects.get(nombre=materia_nombre)

            # Recuperar los temas asociados a la materia
            temas = materia.tema.all()
            temarios = ', '.join([tema.nombre for tema in temas])

            # Recuperar preguntas previas de la materia con embeddings
            preguntas_existentes = Pregunta.objects.filter(materia=materia)
            historial_preguntas = [
                {
                    "pregunta": pregunta.texto_pregunta,
                    "embedding": pregunta.embedding
                }
                for pregunta in preguntas_existentes if pregunta.embedding
            ]

            # Construcción del prompt
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    f"""Eres un profesor experto en la evaluación PAES Chile. Los alumnos vienen a solicitarte preguntas de 5 alternativas, 
                    con alta dificultad. No repitas preguntas similares al historial proporcionado. Responde en formato JSON únicamente.
                    [
                        
                            "pregunta": "Texto de la pregunta",
                            "tema": "Tema correspondiente",
                            "opciones": [
                                "texto": "Opción 1", "es_correcta": null,
                                "texto": "Opción 2", "es_correcta": null,
                                "texto": "Opción 3", "es_correcta": null,
                                "texto": "Opción 4", "es_correcta": null,
                                "texto": "Opción 5", "es_correcta": null
                            ]
                        ,
                        ...
                    ]
                    """),
                ("human",
                    f"""Dame {cantidad} preguntas en formato JSON de los siguientes temarios: {temarios}.
                    """),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            # Configuración del modelo y agente
            seed = int(time.time())
            model = ChatOpenAI(temperature=0, model="gpt-4", seed=seed)
            agent = create_openai_tools_agent(model, TOOLS, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)

            # Ejecutar consulta al agente
            response = agent_executor.invoke({})
            output_text = response.get("output", "")

            # Procesar el JSON recibido
            clean_output = output_text.replace('```json\n', '').replace('```', '')
            try:
                preguntas_data = json.loads(clean_output)
            except json.JSONDecodeError:
                return JsonResponse({"error": "El modelo devolvió un JSON inválido."}, status=500)

            preguntas_guardadas = []
            preguntas_similares = []

            # Instanciamos el modelo de Sentence-Transformer
            model_embed = SentenceTransformer('all-MiniLM-L6-v2')

            # Guardar preguntas en la base de datos
            for pregunta_data in preguntas_data:
                es_similar = False
                similitud_maxima = 0
                pregunta_similar = None

                if historial_preguntas:
                    for historial in historial_preguntas:
                        similitud = similarity_sentence_transformers(
                            pregunta_data["pregunta"], historial["pregunta"]
                        )
                        if similitud > similitud_maxima:
                            similitud_maxima = similitud
                            pregunta_similar = historial["pregunta"]

                if similitud_maxima > 0.85:  # Umbral de similitud
                    es_similar = True
                    preguntas_similares.append({
                        "pregunta_generada": pregunta_data["pregunta"],
                        "pregunta_similar": pregunta_similar,
                        "similitud": similitud_maxima
                    })

                tema = Tema.objects.get(nombre=pregunta_data["tema"])

                # Generar embedding y guardar la pregunta
                nueva_pregunta_embedding = model_embed.encode(pregunta_data["pregunta"]).tolist()
                pregunta = Pregunta.objects.create(
                    texto_pregunta=pregunta_data["pregunta"],
                    materia=materia,
                    tema=tema,
                    embedding=nueva_pregunta_embedding
                )

                # Guardar las opciones de respuesta
                for opcion in pregunta_data["opciones"]:
                    Respuesta.objects.create(
                        texto_respuesta=opcion["texto"],
                        es_correcta=opcion["es_correcta"],
                        pregunta=pregunta
                    )

                # Construir JSON para las preguntas guardadas
                respuestas = [
                    {
                        "id": respuesta.id,
                        "texto_respuesta": respuesta.texto_respuesta,
                        "es_correcta": respuesta.es_correcta,
                    }
                    for respuesta in pregunta.respuestas.all()
                ]

                pregunta_json = {
                    "id": pregunta.id,
                    "texto_pregunta": pregunta.texto_pregunta,
                    "materia": materia_nombre,
                    "tema": tema.nombre,
                    "respuestas": respuestas,
                    "es_similar": es_similar  # Indicar si la pregunta fue marcada como similar
                }

                preguntas_guardadas.append(pregunta_json)

            # Guardar las preguntas generadas en la sesión
            request.session['preguntas_guardadas'] = preguntas_guardadas
            request.session['preguntas_similares'] = preguntas_similares

            print("Preguntas Similares:")
            print(preguntas_similares)
            print("Preguntas JSON:")
            print(preguntas_guardadas)

            # Renderizar la plantilla con las preguntas
            return render(
                request, 
                'pregunta.html', 
                {'preguntas_json': json.dumps(preguntas_guardadas), 'preguntas_similares': preguntas_similares}
            )

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Método de solicitud no permitido."}, status=405)


def historial_usuario(request):
    try:
        # Verificar cuestionarios en la base de datos
        cuestionarios = Cuestionario.objects.all().order_by('-fecha_creacion')

        
        # Obtener datos crudos antes de convertir a lista
        datos_crudos = list(cuestionarios.values())
        
        # Ahora convertimos a lista para la iteración
        cuestionarios = list(cuestionarios)
        
        
        # Serialización
        serializer = CuestionarioSerializer(cuestionarios, many=True)
        datos_serializer = serializer.data
        
        context = {
            'cuestionarios': datos_serializer,
            'hay_datos': bool(datos_serializer),
            'cantidad': len(datos_serializer),
            'datos_crudos': datos_crudos  # Usamos los datos crudos que obtuvimos antes
        }
        
        return render(request, 'historial.html', context)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def retro(request):
    if request.method == 'POST':
        try:
            # Extraer los parámetros de la solicitud
            pregunta = request.POST.get("pregunta")
            respuesta_usuario = request.POST.get("respuesta_usuario")
            respuesta_correcta = request.POST.get("respuesta_correcta")
            materia = request.POST.get("materia")
            # Definir el prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        f"""Eres un profesor experto en {materia} para la evaluación PAES de Chile, donde los alumnos te preguntan porque la respuesta que ellos entregaron es errónea, guiando al estudiante cómo llegar a la respuesta correcta."""
                    ),
                    (
                        "human",
                        f"¿Por qué de la pregunta: {pregunta}, con mi respuesta {respuesta_usuario} es errónea? siendo la respuesta: {respuesta_correcta}."
                    ),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )


            seed = int(time.time())
            model = ChatOpenAI(temperature=0.3, model="gpt-4o", seed=seed)
            agent = create_openai_tools_agent(model, TOOLS, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)
            response = agent_executor.invoke({})
            
            output_text = response.get("output", "")

            return render(request, 'pregunta.html', {"output_text": output_text})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Método de solicitud no permitido."}, status=405)


@csrf_exempt
def comentario_cuestionario(request):
    print("ENTRANDO A LA VISTA")
    if request.method == 'POST':
        print("ENTRANDO AL POST")
        try:
            data = json.loads(request.body)
            cuestionario_id = data.get("cuestionario_id")
            cuestionario = Cuestionario.objects.get(id=cuestionario_id)

            # Agrupar errores por tema
            respuestas_usuario = cuestionario.respuestas_usuario.all()
            errores_por_tema = Counter()

            for respuesta in respuestas_usuario:
                if not respuesta.es_correcta and respuesta.pregunta.tema:
                    tema_nombre = respuesta.pregunta.tema.nombre
                    errores_por_tema[tema_nombre] += 1

            # Generar texto del resumen de errores por tema
            errores_texto = "\n".join(
                f"Tema: {tema}, Errores: {cantidad}"
                for tema, cantidad in errores_por_tema.items()
            )
            print("Errores por tema:", errores_texto)

            # Generar prompt basado en errores por tema
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        f"""Respecto a informacion proporcionada, entrega un resumen básico y comprensible.
                        """),
                    ("human",
                        f"Resumen de errores por tema:\n{errores_texto}\n\n"
                        "¿Cuales son las areas mas debiles y fuertes que tengo?"
                    ),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            seed = int(time.time())
            model = ChatOpenAI(temperature=0.5, model="gpt-4o", seed=seed)
            agent = create_openai_tools_agent(model, TOOLS, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=False)
            response = agent_executor.invoke({})
            output_text = response.get("output", "")

            return JsonResponse({"output": output_text})

        except Exception as e:
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Método de solicitud no permitido."}, status=405)


@csrf_exempt
def obtener_progreso(request):
    if request.method == 'GET':
        try:
            materia = request.GET.get('materia')
            
            # Obtener cuestionarios de la materia específica
            cuestionarios = Cuestionario.objects.filter(
                materia=materia,
                fecha_creacion__isnull=False
            ).order_by('fecha_creacion')
            
            # Diccionario para almacenar resultados por tema
            resultados_por_tema = {}
            fechas = []
            
            for cuestionario in cuestionarios:
                fecha = cuestionario.fecha_creacion.strftime('%Y-%m-%d')
                fechas.append(fecha)
                
                # Obtener respuestas correctas por tema
                temas_correctos = defaultdict(int)
                for respuesta in cuestionario.respuestas_usuario.filter(es_correcta=True):
                    tema = respuesta.pregunta.tema.nombre
                    temas_correctos[tema] += 1
                
                # Guardar resultados por tema
                for tema, correctas in temas_correctos.items():
                    if tema not in resultados_por_tema:
                        resultados_por_tema[tema] = []
                    resultados_por_tema[tema].append(correctas)
                
                # Asegurar que todos los temas tengan un valor para cada fecha
                for tema in resultados_por_tema:
                    if len(resultados_por_tema[tema]) < len(fechas):
                        resultados_por_tema[tema].append(0)
            
            return JsonResponse({
                'valores': list(range(21)),
                'fechas': fechas,
                'resultados_por_tema': resultados_por_tema
            })
            
        except Exception as e:
            print(f"Error en obtener_progreso: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return JsonResponse({'error': str(e)}, status=500)



def get_chart(request, materia):
    # Filtrar los cuestionarios de la materia "Biología"
    cuestionarios = Cuestionario.objects.filter(
        materia='Química',
        fecha_creacion__isnull=False
    ).order_by('fecha_creacion')

    if not cuestionarios.exists():
        return JsonResponse({'error': 'No hay cuestionarios disponibles para la materia seleccionada.'}, status=404)

    # Eje X: Fechas de creación de los cuestionarios
    xAxis_data = [cuestionario.fecha_creacion.strftime('%Y-%m-%d') for cuestionario in cuestionarios]
    print(xAxis_data)

    # Inicializar datos para el gráfico
    temas = set()  # Para rastrear todos los temas únicos
    series_data = {}

    for cuestionario in cuestionarios:
        for pregunta in cuestionario.preguntas.all():
            tema_nombre = pregunta.tema.nombre if pregunta.tema else "Sin tema"
            temas.add(tema_nombre)

            # Contar respuestas incorrectas
            incorrectas = cuestionario.respuestas_usuario.filter(
                pregunta=pregunta, es_correcta=False
            ).count()

            correctas = cuestionario.respuestas_usuario.filter(
                pregunta=pregunta, es_correcta=True
            ).count()

            if tema_nombre not in series_data:
                series_data[tema_nombre] = []

            series_data[tema_nombre].append(incorrectas)

    # Asegurar que cada serie tenga la misma longitud que el eje X
    for tema in temas:
        if tema not in series_data:
            series_data[tema] = [0] * len(xAxis_data)
        else:
            # Rellenar con ceros si falta información
            while len(series_data[tema]) < len(xAxis_data):
                series_data[tema].append(0)

    # Formatear los datos para las series en ECharts
    formatted_series = []
    for tema, data in series_data.items():
        formatted_series.append({
            'name': tema,
            'type': 'line',
            'stack': 'Total',
            'areaStyle': {},
            'emphasis': {
                'focus': 'series'
            },
            'data': data
        })

    # Crear el JSON del gráfico
    option = {
        'title': {
            'text': 'Preguntas Erróneas por Cuestionarios'
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'cross',
                'label': {
                    'backgroundColor': '#6a7985'
                }
            }
        },
        'legend': {
            'data': list(temas)
        },
        'toolbox': {
            'feature': {
                'saveAsImage': {}
            }
        },
        'grid': {
            'left': '3%',
            'right': '4%',
            'bottom': '3%',
            'containLabel': True
        },
        'xAxis': [
            {
                'type': 'category',
                'boundaryGap': False,
                'data': xAxis_data
            }
        ],
        'yAxis': [
            {
                'type': 'value'
            }
        ],
        'series': formatted_series
    }

    return JsonResponse(option)


@csrf_exempt
def preguntas_faltantes(cantidad, temarios):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
                f"""Eres un profesor experto en la evaluación PAES Chile. Los alumnos vienen a solicitarte preguntas de 5 alternativas, 
                con alta dificultad. No repitas preguntas similares al historial proporcionado. Responde en formato JSON únicamente.
                [
                "pregunta": "Texto de la pregunta",
                "tema": "Tema correspondiente",
                "opciones": [
                    "texto": "Opción 1", "es_correcta": null,
                    "texto": "Opción 2", "es_correcta": null,
                    "texto": "Opción 3", "es_correcta": null,
                    "texto": "Opción 4", "es_correcta": null,
                    "texto": "Opción 5", "es_correcta": null
                ]
            ,...
            ]"""
        ),
            ("human",
                f"""Dame {cantidad} preguntas en formato JSON de los siguientes temarios: {temarios}.
                """),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Configuración del modelo y agente
    seed = int(time.time())
    model = ChatOpenAI(temperature=0, model="gpt-4", seed=seed)
    agent = create_openai_tools_agent(model, TOOLS, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)

    # Ejecución del agente
    response = agent_executor.invoke({})  # Ejecutar consulta al agente
    output_text = response.get("output", "")

    # Limpiar y procesar el JSON recibido
    clean_output = output_text.replace('```json\n', '').replace('```', '')

    return clean_output


@csrf_exempt
def get_barra(request, materia):
    try:
        # Filtrar los cuestionarios por materia
        cuestionarios = Cuestionario.objects.filter(materia=materia)

        # Obtener los temas relacionados con la materia
        temas = Tema.objects.filter(materia__nombre=materia).distinct()
        tema_nombres = [tema.nombre for tema in temas]

        # Construir el dataset para el gráfico
        dimensiones = ['cuestionario'] + tema_nombres
        source = []

        for cuestionario in cuestionarios:
            # Inicializar datos del cuestionario con 0 para cada tema
            data_cuestionario = {'cuestionario': cuestionario.titulo}
            for tema_nombre in tema_nombres:
                data_cuestionario[tema_nombre] = 0  # Inicializamos en 0

            # Calcular respuestas correctas seleccionadas por los usuarios para cada tema
            for tema in temas:
                # Filtrar preguntas relacionadas con el tema
                preguntas_tema = cuestionario.preguntas.filter(tema=tema)

                # Filtrar respuestas seleccionadas por el usuario para estas preguntas y que sean correctas
                correctas_tema = cuestionario.respuestas_usuario.filter(
                    pregunta__in=preguntas_tema,
                    es_correcta=True
                ).count()

                # Guardar el conteo de respuestas correctas en el tema
                data_cuestionario[tema.nombre] = correctas_tema
            
            source.append(data_cuestionario)

        # Crear la estructura final
        data = {
            'dataset': {
                'dimensions': dimensiones,
                'source': source,
            }
        }

        return JsonResponse(data)
    
    except Exception as e:
        print(f"Error en get_barra: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def crear_preguntas_retro(request):
    print('Iniciando generación de preguntas')
    if request.method == 'POST':
        try:
            # Extraer los parámetros de la solicitud
            cantidad = int(request.POST.get("cantidad"))
            materia_nombre = request.POST.get("materia")
            materia = Materia.objects.get(nombre=materia_nombre)

            # Recuperar los temas asociados a la materia
            temas = materia.tema.all()

            # Recopilar cuestionarios de la materia
            cuestionarios = Cuestionario.objects.filter(materia=materia_nombre)

            # Calcular porcentaje de aprobación por tema
            porcentaje_aprobacion_por_tema = {}

            for tema in temas:
                total_preguntas = 0
                preguntas_aprobadas = 0

                for cuestionario in cuestionarios:
                    for respuesta in cuestionario.respuestas_usuario.all():
                        if respuesta.pregunta.tema == tema:
                            total_preguntas += 1
                            if respuesta.es_correcta:
                                preguntas_aprobadas += 1

                # Si hay preguntas en el tema, calculamos el porcentaje de aprobación
                if total_preguntas > 0:
                    porcentaje_aprobacion = (preguntas_aprobadas / total_preguntas) * 100
                    porcentaje_aprobacion_por_tema[tema.nombre] = porcentaje_aprobacion
                else:
                    # Si no hay preguntas en el tema, asignamos un 0% de aprobación
                    porcentaje_aprobacion_por_tema[tema.nombre] = 0
            
            # Construcción del prompt priorizando los temas con menor porcentaje de aprobación
            temarios = ', '.join([tema.nombre for tema in temas])

            # Ordenar el diccionario por valores (porcentajes)
            temas_ordenados = sorted(porcentaje_aprobacion_por_tema.items(), key=lambda x: x[1])

            # Obtener los dos temas con menor porcentaje
            dos_menores = temas_ordenados[:2]
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    f"""Eres un profesor experto en la evaluación PAES Chile. Los alumnos vienen a solicitarte preguntas de 5 alternativas, 
                    con alta dificultad. No repitas preguntas similares al historial proporcionado. Responde en formato JSON únicamente.
                    [
                        "pregunta": "Texto de la pregunta",
                        "tema": "Tema correspondiente",
                        "opciones": [
                            "texto": "Opción 1", "es_correcta": null,
                            "texto": "Opción 2", "es_correcta": null,
                            "texto": "Opción 3", "es_correcta": null,
                            "texto": "Opción 4", "es_correcta": null,
                            "texto": "Opción 5", "es_correcta": null
                        ]
                        ,
                        ...
                    ]
                    """),
                ("human",
                    f"""Dame {cantidad} preguntas en formato JSON de todos estos temarios: {temarios}. Dame las preguntas exclusivamente sobre de los 2 temarios con menor porcentaje de aprobacion, en caso de ser todos 0, que haya de cada tema: {temas_ordenados}.
                    """),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            # Configuración del modelo y agente
            seed = int(time.time())
            model = ChatOpenAI(temperature=0, model="gpt-4", seed=seed)
            agent = create_openai_tools_agent(model, TOOLS, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)

            # Ejecutar consulta al agente
            response = agent_executor.invoke({})
            output_text = response.get("output", "")

            # Procesar el JSON recibido
            clean_output = output_text.replace('```json\n', '').replace('```', '')
            print(json.loads(clean_output))
            print(clean_output)
            try:
                preguntas_data = json.loads(clean_output)
            except json.JSONDecodeError:
                return JsonResponse({"error": "El modelo devolvió un JSON inválido."}, status=500)

            preguntas_guardadas = []

            # Instanciamos el modelo de Sentence-Transformer
            model_embed = SentenceTransformer('all-MiniLM-L6-v2')

            # Guardar preguntas en la base de datos
            for pregunta_data in preguntas_data:
                tema = Tema.objects.get(nombre=pregunta_data["tema"])

                # Generar embedding y guardar la pregunta
                nueva_pregunta_embedding = model_embed.encode(pregunta_data["pregunta"]).tolist()
                pregunta = Pregunta.objects.create(
                    texto_pregunta=pregunta_data["pregunta"],
                    materia=materia,
                    tema=tema,
                    embedding=nueva_pregunta_embedding
                )

                # Guardar las opciones de respuesta
                for opcion in pregunta_data["opciones"]:
                    Respuesta.objects.create(
                        texto_respuesta=opcion["texto"],
                        es_correcta=opcion["es_correcta"],
                        pregunta=pregunta
                    )

                # Construir JSON para las preguntas guardadas
                respuestas = [
                    {
                        "id": respuesta.id,
                        "texto_respuesta": respuesta.texto_respuesta,
                        "es_correcta": respuesta.es_correcta,
                    }
                    for respuesta in pregunta.respuestas.all()
                ]

                pregunta_json = {
                    "id": pregunta.id,
                    "texto_pregunta": pregunta.texto_pregunta,
                    "materia": materia_nombre,
                    "tema": tema.nombre,
                    "respuestas": respuestas,
                }

                preguntas_guardadas.append(pregunta_json)

            # Guardar las preguntas generadas en la sesión
            request.session['preguntas_guardadas'] = preguntas_guardadas

            # Renderizar la plantilla con las preguntas
            print("Creacion de prommpt")
            print(prompt)
            print("Cantidad")
            print(cantidad)
            print("Temarios mas bajos")
            print(dos_menores)
            return render(
                request, 
                'pregunta.html', 
                {'preguntas_json': json.dumps(preguntas_guardadas)}
            )

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Método de solicitud no permitido."}, status=405)
