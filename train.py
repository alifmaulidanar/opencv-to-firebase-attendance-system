import os
import io
import cv2
import ctypes
import numpy as np
from PIL import Image
from firebase_admin import firestore
from google.cloud import storage

script_dir = os.path.dirname(os.path.abspath(__file__))

# Path ke file service account Firebase Admin SDK
service_account_path = os.path.join(script_dir, "service_account.json")

# Inisialisasi Firestore
db = firestore.client()

def train_classifier(mahasiswa):
    # Mendapatkan data gambar wajah dari Firebase Cloud Storage
    client = storage.Client.from_service_account_json(service_account_path)
    bucket_name = "metpen-face-recognition.appspot.com"
    bucket = client.get_bucket(bucket_name)
    blobs = mahasiswa

    faces = []
    ids = []

    for blob in blobs:
        blob_name = blob.name
        if blob_name.endswith(".jpg"):
            # Memisahkan nama folder dan nama file
            folder_name, file_name = blob_name.split("/")[-2:]

            # Memisahkan nim dan img_index dari nama folder
            nim, img_index = folder_name.split("_")

            # Hanya memproses file yang sesuai dengan format yang diinginkan
            if file_name.startswith(f"{nim}.{img_index}."):
                # Mendownload data gambar sebagai byte array
                blob_data = blob.download_as_bytes()

                # Mengonversi byte array menjadi gambar
                img = cv2.imdecode(np.frombuffer(blob_data, np.uint8), cv2.IMREAD_GRAYSCALE)

                # Menambahkan gambar ke daftar faces
                faces.append(img)

                # Mendapatkan img_id dari nama file
                id = int(file_name.split(".")[1])
                print(id)

                # Menambahkan id ke daftar ids
                ids.append(id)

                cv2.imshow("Training...", img)
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
    # mahasiswa_ref = db.collection("mahasiswa").document(nim)
    # mahasiswa_ref.update({"face_registered": True})

    ctypes.windll.user32.MessageBoxW(0, "Training Dataset Completed!!", "Result", 1)