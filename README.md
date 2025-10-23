# Portafolio Interactivo con IA (RAG)

Este proyecto es un asistente de IA conversacional que responde preguntas sobre mi perfil profesional. Utiliza un enfoque de **Generación Aumentada por Recuperación (RAG)** para basar sus respuestas en documentos locales como CVs, descripciones de proyectos y otros archivos de texto.

## Arquitectura del Proyecto

El sistema se compone de varios servicios orquestados a través de Docker, diseñados para ser modulares y eficientes:

1.  **Ingesta de Datos (`ingestion_flow.py`)**:
    *   Un flujo de [Prefect](https://www.prefect.io/) que se encarga de leer documentos desde el directorio local `data_local/`.
    *   Utiliza [LangChain](https://www.langchain.com/) para cargar y dividir los documentos en fragmentos (`chunks`).
    *   Genera embeddings (vectores numéricos) para cada fragmento utilizando el modelo `text-embedding-004` de Google Gemini.
    *   Almacena estos vectores en una base de datos vectorial **Qdrant**.

2.  **Base de Datos Vectorial (`qdrant`)**:
    *   Una instancia de [Qdrant](https://qdrant.tech/) que almacena los vectores de los documentos. Actúa como la "memoria" a largo plazo del sistema, permitiendo búsquedas de similitud semántica.

3.  **API de RAG (`rag_api.py`)**:
    *   Una API construida con **FastAPI** que expone un endpoint `/ask` para recibir preguntas.
    *   Recibe una consulta, la convierte en un vector y busca los documentos más relevantes en Qdrant.
    *   Envía la pregunta y los documentos recuperados a un LLM (configurable, por ejemplo, Groq, Google Gemini o OpenAI) para generar una respuesta natural y contextualizada.

4.  **Panel de Administración (`admin_panel.py`)**:
    *   Una aplicación de **Streamlit** que proporciona una interfaz gráfica de usuario para interactuar con la API de RAG de forma visual y amigable.

## Stack Tecnológico

-   **Backend**: Python, FastAPI, LangChain, Prefect, Streamlit
-   **IA & Machine Learning**: Google Generative AI (para Embeddings y LLM), Groq, LangChain
-   **Base de Datos**: Qdrant
-   **Orquestación**: Docker, Docker Compose

## Instalación y Puesta en Marcha

### Prerrequisitos

-   [Docker](https://www.docker.com/get-started) y Docker Compose instalados.
-   Git instalado.
-   Credenciales de API para los servicios de LLM que desees utilizar (Google, Groq, etc.).

### 1. Clonar el Repositorio

```bash
git clone https://github.com/ale-uy/Tu_CV_Chatero.git
cd Tu_CV_Chatero
```

### 2. Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto. Puedes copiar el archivo `.env.example` como plantilla:

```bash
cp .env.example .env
```

Ahora, edita el archivo `.env` y completa las variables con tus propias credenciales y configuraciones. Es crucial que `COLLECTION_NAME` sea el mismo para la ingesta y la API.

### 3. Colocar los Documentos

Agrega los archivos (`.pdf`, `.md`, `.docx`, etc.) que conformarán la base de conocimiento de la IA dentro del directorio `data_local/` y sus subcarpetas (`CV/`, `proyectos/` y `repos/`).

### 4. Iniciar los Servicios

El proyecto utiliza perfiles de Docker Compose para separar el entorno principal del proceso de ingesta.

**a) Iniciar los servicios principales (API, Qdrant, Admin Panel):**

```bash
docker-compose up --build -d
```

**b) Ejecutar la ingesta de datos:**

Una vez que los servicios principales estén en funcionamiento, abre otra terminal y ejecuta el perfil de ingesta. Esto procesará los archivos en `data_local/` y los cargará en Qdrant.

```bash
docker-compose up --profile ingest --build
```

Este comando iniciará el contenedor `ingestor`, ejecutará el script `ingestion_flow.py` y se detendrá una vez completado.

## Uso

### Panel de Administración

-   Accede a la interfaz de Streamlit en tu navegador: `http://localhost:8501`

### API de RAG

-   La API estará disponible en `http://localhost:8000`.
-   Puedes ver tus colecciones Qdrant en `http://localhost:3000/dashboard`.
-   Puedes enviar preguntas al endpoint `/ask` usando `curl` o cualquier cliente HTTP:

```bash
curl -X POST "http://localhost:8000/ask" \
-H "Content-Type: application/json" \
-d '{"query": "¿Cuál es tu experiencia con proyectos de Machine Learning?"}'
```

-   Consulta la documentación interactiva de la API (generada por FastAPI) en: `http://localhost:8000/docs`

### EN PRODUCCION: [CHATEA CONMIGO](https://ale-uy.github.io)
