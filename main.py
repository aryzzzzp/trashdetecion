from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# --- KONFIGURASI ---
# Izinkan frontend mengakses backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
# Pastikan file model_CNN.keras ada di folder yang sama
try:
    model = tf.keras.models.load_model("model_weights.h5")
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error memuat model: {e}")

# Kelas Sampah (Sesuaikan urutan alfabetis folder training kamu)
# Biasanya: 0 = Anorganik, 1 = Organik
CLASS_NAMES = ["Anorganik", "Organik"]

def preprocess_image(image_bytes):
    """Fungsi untuk memproses gambar agar sesuai input model"""
    # 1. Buka gambar dari bytes
    img = Image.open(io.BytesIO(image_bytes))
    
    # 2. Ubah ukuran ke 128x128 (Sesuai training kamu)
    img = img.resize((128, 128))
    
    # 3. Ubah ke array numpy
    img_array = np.array(img)
    
    # 4. Pastikan RGB (jika ada gambar grayscale/PNG transparan)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    # 5. Expand dims (Menambah dimensi batch: 1, 128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 6. Rescaling / Normalisasi (0-255 menjadi 0-1)
    # PENTING: Karena di training kamu melakukan rescaling di dataset.map,
    # maka di sini kita harus manual membagi 255.
    img_array = img_array / 255.0
    
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Baca file gambar
    image_bytes = await file.read()
    
    # Preprocess
    processed_image = preprocess_image(image_bytes)
    
    # Prediksi
    predictions = model.predict(processed_image)
    score = tf.nn.softmax(predictions[0]) # Hitung probabilitas
    
    # Ambil kelas dengan nilai tertinggi
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0])) * 100

    return {
        "class": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)