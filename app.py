from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI(title="API de Visión - Hola Mundo")

# 1. Cargar modelo pre-entrenado (MobileNet V2 es ligero y rápido)
# Cuando los alumnos tengan su propio modelo, cambiarán esta línea por:
# model = torch.load("mi_modelo_entrenado.pt")
modelo = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
modelo.eval()

# 2. Definir las transformaciones de imagen que exige el modelo
transformacion = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/")
def estado_servidor():
    return {"estado": "Servidor activo. Usa el endpoint /predecir/ para enviar fotos."}

@app.post("/predecir/")
async def predecir_imagen(archivo: UploadFile = File(...)):
    # Leer la imagen enviada por la App Móvil
    datos_imagen = await archivo.read()
    imagen = Image.open(io.BytesIO(datos_imagen)).convert('RGB')
    
    # Transformar la imagen a formato tensor numérico
    tensor = transformacion(imagen).unsqueeze(0)

    # Ejecutar la predicción sin calcular gradientes (ahorra memoria)
    with torch.no_grad():
        prediccion = modelo(tensor)

    # Obtener el número de clase con mayor probabilidad
    _, indice_ganador = torch.max(prediccion, 1)
    
    return {
        "nombre_archivo": archivo.filename,
        "clase_predicha": indice_ganador.item(),
        "mensaje": "Reemplaza esta clase numérica con tus nombres en el futuro."
    }