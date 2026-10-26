"""
API de reconocimiento facial - EP2 Deep Learning (AVY1103)
Basada en la plantilla trigoduoc/proyecto-api-vision.
Modelo: CNN propia entrenada en Colab con rostros recortados (modelo/modelo_rostros.keras).
Preprocesamiento idéntico al entrenamiento: detección de rostro (YuNet) -> recorte con margen -> 128x128x3.
"""
import io
import json
from pathlib import Path

import cv2
import keras
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

CARPETA_MODELO = Path(__file__).parent / "modelo"
configuracion = json.loads((CARPETA_MODELO / "configuracion_modelo.json").read_text(encoding="utf-8"))
TAMANO_IMAGEN = configuracion["tamano_imagen"][0]          # 128
UMBRAL_CONFIANZA = configuracion["umbral_confianza"]       # 0.6
MARGEN_ROSTRO = configuracion["margen_rostro"]             # 0.30
CLASES = configuracion["clases"]                           # índice i de la salida softmax -> CLASES[i]
LADO_DETECCION = 640

# Se cargan una sola vez al iniciar el servidor
modelo = keras.models.load_model(CARPETA_MODELO / "modelo_rostros.keras")
detector_rostros = cv2.FaceDetectorYN.create(
    str(CARPETA_MODELO / configuracion["detector_rostros"]), "", (320, 320), score_threshold=0.6)

app = FastAPI(
    title="API de Reconocimiento Facial - Deep Learning DuocUC",
    description="Recibe una foto (128x128x3 desde la app), detecta el rostro y responde a qué persona del "
                "curso corresponde, o indica que no está en la base de datos si la confianza es menor al umbral.",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def recortar_rostro(imagen: Image.Image):
    """Mismo recorte que en Colab (Etapa 10b): cara más grande + 30% de margen, cuadrado dentro de la imagen.
    Devuelve (matriz uint8 128x128x3, se_detecto_rostro)."""
    ancho, alto = imagen.size
    escala = LADO_DETECCION / max(ancho, alto)
    imagen_deteccion = imagen.resize((int(ancho * escala), int(alto * escala)))
    matriz_bgr = cv2.cvtColor(np.asarray(imagen_deteccion), cv2.COLOR_RGB2BGR)
    detector_rostros.setInputSize((matriz_bgr.shape[1], matriz_bgr.shape[0]))
    _, rostros = detector_rostros.detect(matriz_bgr)
    if rostros is None or len(rostros) == 0:
        return None, False

    x, y, w, h = max(rostros, key=lambda r: r[2] * r[3])[:4]
    centro_x, centro_y = (x + w / 2) / escala, (y + h / 2) / escala
    lado = int(min(max(w, h) / escala * (1 + 2 * MARGEN_ROSTRO), ancho, alto))
    izquierda = int(np.clip(centro_x - lado / 2, 0, ancho - lado))
    arriba = int(np.clip(centro_y - lado / 2, 0, alto - lado))
    recorte = imagen.crop((izquierda, arriba, izquierda + lado, arriba + lado))
    recorte = recorte.resize((TAMANO_IMAGEN, TAMANO_IMAGEN), Image.BILINEAR)
    return np.asarray(recorte, dtype=np.uint8), True


@app.get("/", summary="Estado del servidor")
def estado_servidor():
    return {"estado": "Servidor activo. Usa el endpoint /predecir/ para enviar fotos.",
            "personas_registradas": len(CLASES), "umbral_confianza": UMBRAL_CONFIANZA,
            "metricas_prueba": configuracion["metricas_prueba"]}


@app.get("/personas/", summary="Lista de personas que el modelo reconoce")
def listar_personas():
    return {"total": len(CLASES), "personas": [c["nombre"] for c in CLASES]}


@app.post("/predecir/", summary="Reconocer a la persona de una foto")
async def predecir_imagen(archivo: UploadFile = File(..., description="Foto del rostro 128x128x3 (.jpg o .png)")):
    datos_imagen = await archivo.read()
    try:
        with Image.open(io.BytesIO(datos_imagen)) as imagen_recibida:
            imagen = ImageOps.exif_transpose(imagen_recibida).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="El archivo enviado no es una imagen válida.")

    rostro, detectado = recortar_rostro(imagen)
    if not detectado:
        return {"nombre_archivo": archivo.filename, "reconocido": False, "nombre": None, "id_carpeta": None,
                "confianza": 0.0, "rostro_detectado": False,
                "mensaje": "No se detectó un rostro en la foto. Acerca la cámara y centra la cara.", "top3": []}

    probabilidades = modelo(rostro[np.newaxis].astype(np.float32), training=False).numpy()[0]
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
        "rostro_detectado": True,
        "mensaje": (f"Persona reconocida: {CLASES[indice_ganador]['nombre']}" if reconocido
                    else "La persona no está en la base de datos."),
        "top3": [{"nombre": CLASES[int(i)]["nombre"], "probabilidad": round(float(probabilidades[i]), 4)}
                 for i in indices_top3],
    }