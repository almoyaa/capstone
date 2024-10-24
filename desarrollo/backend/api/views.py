import os
import json
import openai
from django.shortcuts import render
from rest_framework import generics
from .models import Usuario, Cuestionario, Pregunta, Respuesta, Materia, Tema
from .serializers import UsuarioSerializer, CuestionarioSerializer, PreguntaSerializer, MateriaSerializer
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.http import JsonResponse
from dotenv import load_dotenv
from django.views.generic import TemplateView
from django.shortcuts import redirect
from django.utils.safestring import mark_safe

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.schema import Document  
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent

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
    
class MateriaListView(generics.ListAPIView):
    queryset = Materia.objects.all()
    serializer_class = MateriaSerializer



client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



class IndexTemplateView(TemplateView):
    template_name = "index.html"

class PreguntaTemplateView(TemplateView):
    template_name = "pregunta.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Cargar preguntas de la sesión
        preguntas = self.request.session.get('preguntas_guardadas', [])
        context['preguntas_json'] = json.dumps(preguntas)  # Convertir a JSON
        print(context)
        return context


class HistorialTemplateView(TemplateView):
    template_name = "historial.html"

class CrearCuestionarioView(TemplateView):
    template_name="crear-cuestionario.html"

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

    print("Documentos cargados")
    # Crear el índice vectorial FAISS con todos los documentos combinados
    faiss_index = FAISS.from_documents(todos_los_documentos, embeddings_model)

    # Retornar el retriever
    return faiss_index.as_retriever()



@csrf_exempt
def crear_pregunta_matematica(request):
    if request.method == 'POST':
        print(request.POST)
        try:
            # Extraer los parámetros de la solicitud
            cantidad = request.POST.get("cantidad")
            materia = Materia.objects.get(nombre=request.POST.get("materia"))
            temas = materia.tema.all()

            # Ruta a la carpeta con los PDFs
            carpeta_pdfs = "C:/Users/Seba/Desktop/Notebooks/Contenidos"

            # Crear embeddings
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            retriever = cargar_pdfs_desde_carpeta(carpeta_pdfs, embeddings)

            # Definir el prompt
            temarios = ', '.join([tema.nombre for tema in temas])
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """Eres un profesor experto en la evaluación PAES Chile, y los alumnos vienen a solicitarte preguntas de 5 alternativas para poner a prueba sus conocimientos. Además de conocer los contenidos del temario de cada evaluación, ya sea, competencias de matemática M1 y matemática M2, solo entrega json, nada más de texto, solo json.
                        
                        [
                {{
                    "pregunta": "Texto de la pregunta",
                    "tema":"tema correspondiente",
                    "opciones": [
                        {{"texto": "Opción 1", "es_correcta": null}},
                        {{"texto": "Opción 2", "es_correcta": null}},
                        {{"texto": "Opción 3", "es_correcta": null}},
                        {{"texto": "Opción 4", "es_correcta": null}},
                        {{"texto": "Opción 5", "es_correcta": null}}
                    ]
                }},
                ...
            ]"""
                    ),
                    ("human", f"Dame {cantidad} preguntas en formato JSON de los siguientes temarios: {temarios}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            # Crear herramientas y ejecutar el agente
            retriever_tool = create_retriever_tool(
                retriever,
                "Temarios_prueba_PAES",
                "Recurso para establecer el contenido con los que se van evaluar los conocimientos del alumno, para cada temario.",
            )

            tools = [retriever_tool]
            model = ChatOpenAI(temperature=0, model="gpt-4o", seed=211)
            agent = create_openai_tools_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            # Ejecutar la consulta
            response = agent_executor.invoke({"input": f"Dame {cantidad} preguntas en formato json de los siguientes temarios: {temarios}, solo json, nada más, SOLO JSON"})
            output_text = response.get("output", "")

            # El JSON limpio
            clean_output = output_text.replace('```json\n', '').replace('```', '')
            preguntas_data = json.loads(clean_output)

            print("ENTRANDO AL FOR")

            preguntas_guardadas = []

            for pregunta_data in preguntas_data:
                # Obtener la materia y el tema
                tema = Tema.objects.get(nombre=pregunta_data["tema"])

                # Crear la pregunta
                pregunta = Pregunta.objects.create(
                    texto_pregunta=pregunta_data["pregunta"],
                    materia=materia,
                    tema=tema
                )

                # Crear las opciones de respuesta
                for opcion in pregunta_data["opciones"]:
                    Respuesta.objects.create(
                        texto_respuesta=opcion["texto"],
                        es_correcta=opcion["es_correcta"],
                        pregunta=pregunta
                    )

                # Preparar los datos de las respuestas
                respuestas = [
                    {
                        "id": respuesta.id,
                        "texto_respuesta": respuesta.texto_respuesta,
                        "es_correcta": respuesta.es_correcta
                    }
                    for respuesta in pregunta.respuestas.all()
                ]

                # Formar el JSON de la pregunta
                pregunta_json = {
                    "id": pregunta.id,
                    "texto_pregunta": pregunta.texto_pregunta,
                    "materia": pregunta.materia.nombre,
                    "tema":pregunta.tema.nombre,
                    "respuestas": respuestas
                }
                print( )

                preguntas_guardadas.append(pregunta_json)

            # Guardar las preguntas en la sesión
            request.session['preguntas_guardadas'] = preguntas_guardadas

            return render(request, 'pregunta.html', {'preguntas_json': mark_safe(json.dumps(preguntas_guardadas))})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Método de solicitud no permitido."}, status=405)
