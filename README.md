# Plantilla API de Visión Computacional

Este repositorio contiene un "Hola Mundo" para servir un modelo de Machine Learning mediante una API REST. Esta API es la que se comunicará con la Aplicación Móvil.

## Pre-requisitos
* Python 3.10 o superior.

## Instalación

Crea un entorno virtual e instala las dependencias:
```bash
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

## Ejecución del Servidor

Levanta la API con uvicorn:
```bash
uvicorn app:app --reload
```
El servidor quedará corriendo en `http://127.0.0.1:8000`.

## Cómo probarlo (Sin necesidad de app móvil aún)

FastAPI incluye una interfaz visual para probar envíos de fotos.
1. Abre en tu navegador: `http://127.0.0.1:8000/docs`
2. Despliega el endpoint verde **POST /predecir/**.
3. Haz clic en el botón "Try it out".
4. Sube cualquier imagen de prueba desde tu computador.
5. Haz clic en "Execute" y revisa la respuesta del servidor (JSON).

## Evolución del Proyecto
Para el proyecto final, deberás:
1. Reemplazar el modelo pre-entrenado de PyTorch en `app.py` por el modelo de reconocimiento facial que entrenaste con los datos del curso.
2. Modificar la respuesta para que devuelva el nombre de la persona en lugar de un número de clase genérico.