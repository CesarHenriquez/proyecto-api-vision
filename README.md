# API de Reconocimiento Facial · EP2 Deep Learning (AVY1103)

API REST que recibe una foto de un rostro (128×128×3) desde la app móvil y responde **a qué persona del curso corresponde**, o indica que **no está en la base de datos**.
Basada en la plantilla [trigoduoc/proyecto-api-vision](https://github.com/trigoduoc/proyecto-api-vision).

**Integrantes:** Benjamin Leal · Cesar Henriquez

## Cómo funciona
1. La app envía la imagen al endpoint `POST /predecir/` (campo `archivo`).
2. La API corrige la rotación EXIF, **detecta el rostro con YuNet** (OpenCV) y lo recorta con 30% de margen, igual que en el entrenamiento.
3. La **CNN propia** (4 bloques Conv + BatchNorm + ReLU + MaxPooling, preentrenada con LFW y ajustada con el dataset del curso) entrega una probabilidad por persona.
4. Si la probabilidad más alta es **≥ 0.5**, responde el nombre; si no, responde "La persona no está en la base de datos".

## Modelo (`modelo/`)
| Archivo | Contenido |
|---|---|
| `modelo_rostros.keras` | CNN entrenada en Google Colab (TensorFlow 2.21 / Keras 3) |
| `configuracion_modelo.json` | 31 personas (en el orden de la salida softmax), umbral y métricas |
| `face_detection_yunet_2023mar.onnx` | Detector de rostros YuNet |

Métricas en el conjunto de prueba: accuracy 80,3 % · precision 79,6 % · recall 80,7 % · F1 78,0 % (promedio macro).

## Instalación
Requiere Python 3.10+.
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Ejecución
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
- `--host 0.0.0.0` permite que el celular se conecte por la red Wi-Fi usando la IP del PC: `http://IP_DEL_PC:8000`.
- Si la red bloquea la conexión entre equipos, se puede publicar con un túnel: `cloudflared tunnel --url http://localhost:8000`, y usar en la app la dirección `https://...trycloudflare.com`.

## Endpoints
Documentación interactiva en `http://localhost:8000/docs`.

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Estado del servidor, umbral y métricas |
| GET | `/personas/` | Lista de las 31 personas que reconoce el modelo |
| POST | `/predecir/` | Recibe la foto (`archivo`) y devuelve la predicción |

Ejemplo de respuesta de `/predecir/`:
```json
{
  "reconocido": true,
  "nombre": "Benjamin Palma",
  "id_carpeta": "bp_01",
  "confianza": 0.97,
  "rostro_detectado": true,
  "mensaje": "Persona reconocida: Benjamin Palma",
  "top3": [
    {"nombre": "Benjamin Palma", "probabilidad": 0.97},
    {"nombre": "Javier Montupil", "probabilidad": 0.01},
    {"nombre": "Nicolas Cuevas", "probabilidad": 0.01}
  ]
}
```

## Repositorios relacionados
- App móvil (Expo): https://github.com/CesarHenriquez/app-vision
- Dataset del curso (Git + DVC): https://github.com/trigoduoc/proyecto-dataset-vision
