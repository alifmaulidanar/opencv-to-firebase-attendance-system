import os
import io
from PIL import Image
import numpy as np
import cv2
import ctypes
from firebase_admin import firestore
from google.cloud import storage

script_dir = os.path.dirname(os.path.abspath(__file__))

# Path ke file service account Firebase Admin SDK
service_account_path = os.path.join(script_dir, "service_account.json")

# Inisialisasi Firestore
db = firestore.client()

def train_classifier(nim, img_index):
    # Mendapatkan data gambar wajah dari Firebase Cloud Storage
    client = storage.Client.from_service_account_json(service_account_path)
    bucket_name = "metpen-face-recognition.appspot.com"
    bucket = client.get_bucket(bucket_name)
    folder_path = f"mahasiswa/{nim}_{img_index}/"
    blob_prefix = f"{folder_path}{nim}."
    blobs = bucket.list_blobs(prefix=blob_prefix)

    faces = []
    ids = []

    for blob in blobs:
        blob_data = blob.download_as_bytes()
        img = Image.open(io.BytesIO(blob_data)).convert("L")
        
        # Konversi ke array numpy
        image_np = np.array(img, 'uint8')
        
        id = int(blob.name.split(".")[1])
        
        faces.append(image_np)
        ids.append(id)
        
        cv2.imshow("Training...", image_np)
        cv2.waitKey(1)
    
    ids = np.array(ids)

    # Training
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)

    # Menyimpan classifier
    classifier_path = "model_classifier.xml"
    clf.write(classifier_path)

    # Mengunggah classifier ke Firebase Cloud Storage
    blob_path = f"model/classifier.xml"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(classifier_path)

    # Menghapus file lokal classifier
    os.remove(classifier_path)

    # Simpan data training status di Firestore
    mahasiswa_ref = db.collection("mahasiswa").document(nim)
    mahasiswa_ref.update({"face_registered": True})

    ctypes.windll.user32.MessageBoxW(0, "Training Dataset Completed!!", "Result", 1)