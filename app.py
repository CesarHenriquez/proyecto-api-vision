"""
API de reconocimiento facial - EP2 Deep Learning (AVY1103)
Basada en la plantilla trigoduoc/proyecto-api-vision.
Sirve la CNN propia entrenada en Colab (modelo/modelo_rostros.keras).
"""
import io
import json
from pathlib import Path

import numpy as np
import keras
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

CARPETA_MODELO = Path(__file__).parent / "modelo"
configuracion = json.loads((CARPETA_MODELO / "configuracion_modelo.json").read_text(encoding="utf-8"))
TAMANO_IMAGEN = configuracion["tamano_imagen"][0]          # 128
UMBRAL_CONFIANZA = configuracion["umbral_confianza"]       # 0.6
CLASES = configuracion["clases"]                           # índice i de la salida softmax -> CLASES[i]

# El modelo se carga una sola vez al iniciar el servidor (no en cada petición)
modelo = keras.models.load_model(CARPETA_MODELO / "modelo_rostros.keras")

app = FastAPI(
    title="API de Reconocimiento Facial - Deep Learning DuocUC",
    description="Recibe una foto de un rostro y responde a qué persona del curso corresponde, "
                "o indica que no está en la base de datos si la confianza es menor al umbral.",
    version="1.0.0",
)
# Permite llamadas desde la app (Expo web / navegador). Las apps nativas no lo necesitan.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def preprocesar_imagen(datos_imagen: bytes) -> np.ndarray:
    """Mismo proceso que en el entrenamiento: EXIF -> RGB -> recorte cuadrado central -> 128x128.
    Devuelve un lote de 1 imagen uint8 de forma (1, 128, 128, 3); la normalización está dentro del modelo."""
    with Image.open(io.BytesIO(datos_imagen)) as imagen:
        imagen = ImageOps.exif_transpose(imagen).convert("RGB")
        lado_menor = min(imagen.size)
        imagen_cuadrada = ImageOps.fit(imagen, (lado_menor, lado_menor), centering=(0.5, 0.5))
        imagen_final = imagen_cuadrada.resize((TAMANO_IMAGEN, TAMANO_IMAGEN), Image.BILINEAR)
        return np.asarray(imagen_final, dtype=np.uint8)[np.newaxis, ...]


@app.get("/", summary="Estado del servidor")
def estado_servidor():
    return {"estado": "Servidor activo. Usa el endpoint /predecir/ para enviar fotos.",
            "personas_registradas": len(CLASES), "umbral_confianza": UMBRAL_CONFIANZA}


@app.get("/personas/", summary="Lista de personas que el modelo reconoce")
def listar_personas():
    return {"total": len(CLASES), "personas": [c["nombre"] for c in CLASES]}


@app.post("/predecir/", summary="Reconocer a la persona de una foto")
async def predecir_imagen(archivo: UploadFile = File(..., description="Foto del rostro (.jpg o .png)")):
    datos_imagen = await archivo.read()
    try:
        lote_imagen = preprocesar_imagen(datos_imagen)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="El archivo enviado no es una imagen válida.")

    probabilidades = modelo(lote_imagen, training=False).numpy()[0]   # 31 probabilidades que suman 1
    indices_top3 = np.argsort(probabilidades)[::-1][:3]
    indice_ganador = int(indices_top3[0])
    confianza = float(probabilidades[indice_ganador])
    reconocido = confianza >= UMBRAL_CONFIANZA

    return {
        "nombre_archivo": archivo.filename,
        "reconocido": reconocido,
        "nombre": CLASES[indice_ganador]["nombre"] if reconocido else None,
        "id_carpeta": CLASES[indice_ganador]["id_carpeta"] if reconocido else None,
        "confianza": round(confianza, 4),
        "mensaje": (f"Persona reconocida: {CLASES[indice_ganador]['nombre']}" if reconocido
                    else "La persona no está en la base de datos."),
        "top3": [{"nombre": CLASES[int(i)]["nombre"], "probabilidad": round(float(probabilidades[i]), 4)}
                 for i in indices_top3],
    }