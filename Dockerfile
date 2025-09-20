# Usar una imagen base de Python más moderna
FROM python:3.11-slim

# Establecer el directorio de trabajo en /app
WORKDIR /app

# Crear un usuario no-root para mejorar la seguridad
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copiar el archivo de dependencias primero para aprovechar el cache de Docker
COPY --chown=user ./requirements.txt requirements.txt

# Instalar las dependencias
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copiar todo el código de la aplicación
COPY --chown=user . .

# Comando para iniciar la API de FastAPI con Uvicorn
# Escuchará en todas las interfaces en el puerto 7860, el estándar de HF Spaces
CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "7860"]
