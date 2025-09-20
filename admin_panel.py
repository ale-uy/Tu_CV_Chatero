import streamlit as st
import requests
import os
import time

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Panel de Control RAG",
    page_icon="🤖",
    layout="wide"
)

# --- Funciones para obtener modelos ---
def get_lm_studio_models(base_url):
    """Obtiene los modelos disponibles desde la API de LM Studio."""
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        response.raise_for_status()
        models = response.json().get("data", [])
        return [model["id"] for model in models]
    except requests.exceptions.RequestException as e:
        st.warning(f"No se pudieron cargar los modelos de LM Studio: {e}")
        return []

def get_groq_models(api_key):
    """Obtiene los modelos disponibles desde la API de Groq."""
    if not api_key:
        return []
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=5)
        response.raise_for_status()
        models = response.json().get("data", [])
        return [model["id"] for model in models]
    except requests.exceptions.RequestException as e:
        st.warning(f"No se pudieron cargar los modelos de Groq. ¿API Key válida? Error: {e}")
        return []

def get_gemini_models(api_key):
    """Obtiene los modelos disponibles desde la API de Gemini."""
    if not api_key:
        return []
    try:
        headers = {"x-goog-api-key": api_key}
        response = requests.get("https://generativelanguage.googleapis.com/v1beta/models", headers=headers, timeout=5)
        response.raise_for_status()
        models_data = response.json().get("models", [])
        supported_models = [
            model["name"].split('/')[-1] 
            for model in models_data 
            if "generateContent" in model.get("supportedGenerationMethods", [])
        ]
        return supported_models
    except requests.exceptions.RequestException as e:
        st.warning(f"No se pudieron cargar los modelos de Gemini. ¿API Key válida? Error: {e}")
        return []

def get_openai_models(api_key):
    """Obtiene los modelos disponibles desde la API de OpenAI."""
    if not api_key:
        return []
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=5)
        response.raise_for_status()
        models = response.json().get("data", [])
        return sorted([model["id"] for model in models])
    except requests.exceptions.RequestException as e:
        st.warning(f"No se pudieron cargar los modelos de OpenAI. ¿API Key válida? Error: {e}")
        return []

# --- Variables de Entorno y URLs ---
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
STATUS_ENDPOINT = f"{RAG_API_URL}/config/status"
CONFIG_ENDPOINT = f"{RAG_API_URL}/config/llm"
ASK_ENDPOINT = f"{RAG_API_URL}/ask"

# --- Estado de la Sesión de Streamlit ---
# Usamos el estado de la sesión para mantener los valores entre interacciones
if 'active_provider' not in st.session_state:
    st.session_state.active_provider = None
if 'active_model' not in st.session_state:
    st.session_state.active_model = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Funciones de la API ---
def get_api_status():
    """Obtiene el estado actual del LLM configurado en la API."""
    try:
        response = requests.get(STATUS_ENDPOINT)
        response.raise_for_status()
        data = response.json()
        st.session_state.active_provider = data.get("active_provider")
        st.session_state.active_model = data.get("active_model")
    except requests.exceptions.RequestException as e:
        st.session_state.active_provider = "error"
        st.error(f"No se pudo conectar a la API RAG en {RAG_API_URL}. ¿Está corriendo? Error: {e}")

def configure_llm(provider, model_name, api_key=None, base_url=None):
    """Envía la configuración del nuevo LLM a la API."""
    config_payload = {
        "provider": provider,
        "model_name": model_name,
        "api_key": api_key,
        "base_url": base_url
    }
    try:
        with st.spinner(f"Configurando {model_name}..."):
            response = requests.post(CONFIG_ENDPOINT, json=config_payload)
            response.raise_for_status()
            get_api_status() # Actualizar el estado después de la configuración
        st.success(f"¡Modelo {st.session_state.active_model} de {st.session_state.active_provider} activado!")
    except requests.exceptions.RequestException as e:
        error_detail = str(e) # Valor por defecto
        if e.response is not None:
            try:
                # Intenta obtener el detalle del JSON, si falla, usa el texto plano
                error_detail = e.response.json().get("detail", e.response.text)
            except requests.exceptions.JSONDecodeError:
                error_detail = e.response.text
        
        st.error(f"Error al configurar el modelo: {error_detail}")

# --- UI del Panel de Administrador ---
st.title("Panel de Control para el Asistente RAG")
st.markdown("Desde aquí puedes configurar, probar y gestionar tu asistente de carrera profesional.")

# Actualizar el estado al cargar la página
get_api_status()

# --- Columnas para la UI ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("⚙️ Configuración del LLM")
    
    # Indicador de estado
    if st.session_state.active_provider and st.session_state.active_provider != "error":
        st.success(f"**Activo:** {st.session_state.active_model} ({st.session_state.active_provider})")
    else:
        st.warning("**Activo:** Ningún LLM configurado.")

    with st.expander("Seleccionar y Activar un Nuevo Modelo", expanded=True):
        provider = st.selectbox(
            "Proveedor del Modelo",
            ("lm_studio", "groq", "gemini", "openai"),
            help="Elige el servicio que proporcionará el modelo de lenguaje."
        )

        model_name = ""
        api_key_input = None
        base_url = None

        if provider == "lm_studio":
            st.info("Asegúrate de que LM Studio esté corriendo y que el servidor esté iniciado.")
            base_url = st.text_input(
                "URL Base del Servidor Local", 
                "http://host.docker.internal:1234", 
                help="URL del servidor de LM Studio. `host.docker.internal` es para acceder al host desde un contenedor Docker."
            )
            with st.spinner("Cargando modelos de LM Studio..."):
                lm_studio_models = get_lm_studio_models(base_url)
            
            if lm_studio_models:
                model_name = st.selectbox("Modelo", lm_studio_models, help="Selecciona un modelo de la lista.")
            else:
                st.warning("No se pudieron cargar los modelos. Introduce el nombre manualmente.")
                model_name = st.text_input("Nombre del Modelo", "local-model", help="El nombre del modelo tal como aparece en LM Studio.")
        
        elif provider in ["groq", "gemini", "openai"]:
            provider_name = provider.capitalize()

            # Determinar la variable de entorno correcta
            if provider == "gemini":
                api_env_var = "GOOGLE_API_KEY"
            else: # groq, openai
                api_env_var = f"{provider.upper()}_API_KEY"

            st.info(f"La clave de API se tomará del archivo .env ({api_env_var}) si no se especifica una aquí.")
            api_key_input = st.text_input(f"Clave de API de {provider_name} (Opcional)", type="password", help=f"Pega tu clave de API de {provider_name}.")
            
            api_key_to_use = api_key_input if api_key_input else os.getenv(api_env_var)
            
            models = []
            with st.spinner(f"Cargando modelos de {provider_name}..."):
                if provider == "groq":
                    models = get_groq_models(api_key_to_use)
                elif provider == "gemini":
                    models = get_gemini_models(api_key_to_use)
                elif provider == "openai":
                    models = get_openai_models(api_key_to_use)

            if models:
                default_model = ""
                if provider == "groq":
                    default_model = "llama3-8b-8192"
                elif provider == "gemini":
                    default_model = "gemini-1.5-flash-latest"
                elif provider == "openai":
                    default_model = "gpt-4o"
                
                sorted_models = sorted(models)
                index = sorted_models.index(default_model) if default_model in sorted_models else 0
                model_name = st.selectbox("Modelo", sorted_models, index=index)
            else:
                st.warning(f"No se pudieron cargar los modelos de {provider_name}. Introduce el nombre manualmente.")
                default_model_name = "llama3-8b-8192" if provider == "groq" else "gemini-1.5-flash-latest" if provider == "gemini" else "gpt-4o"
                model_name = st.text_input("Modelo", default_model_name)

        if st.button("Activar Modelo"):
            if not model_name:
                st.error("El nombre del modelo no puede estar vacío.")
            else:
                configure_llm(provider, model_name, api_key_input, base_url)

    st.header("📦 Ingesta de Datos")
    st.info("La ingesta procesa tus archivos de la carpeta `data_local` y los guarda en la base de datos vectorial para que el bot pueda usarlos.")
    st.warning("**Acción Manual Requerida:** Para ejecutar la ingesta, abre una terminal en la raíz de tu proyecto y ejecuta el siguiente comando:")
    st.code("docker-compose run --rm ingestor", language="bash")

with col2:
    st.header("💬 Chat de Prueba")

    if not st.session_state.active_provider or st.session_state.active_provider == "error":
        st.warning("Debes configurar un LLM en el panel de la izquierda para poder chatear.")
    else:
        # Lógica del chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Pregúntame algo sobre tu carrera..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    with st.spinner("Pensando..."):
                        response = requests.post(ASK_ENDPOINT, json={"query": prompt})
                        response.raise_for_status()
                        answer = response.json().get("answer", "No se recibió respuesta.")
                        sources = response.json().get("sources", [])
                    
                    message_placeholder.markdown(answer)
                    
                    with st.expander("Fuentes consultadas"):
                        for i, source in enumerate(sources):
                            st.info(f"**Fuente {i+1}**\n*Metadata: {source.get('metadata', {})['source'] if source.get('metadata') else 'N/A'}*\n---\n{source['content']}")

                except requests.exceptions.RequestException as e:
                    error_detail = str(e) # Valor por defecto
                    if e.response is not None:
                        try:
                            error_detail = e.response.json().get("detail", e.response.text)
                        except requests.exceptions.JSONDecodeError:
                            error_detail = e.response.text
                    st.error(f"Error al contactar la API: {error_detail}")
                    answer = f"Error: {error_detail}"

            st.session_state.messages.append({"role": "assistant", "content": answer})
