import cv2
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud import storage

script_dir = os.path.dirname(os.path.abspath(__file__))

# Path ke file service account Firebase Admin SDK
service_account_path = os.path.join(script_dir, "service_account.json")

# Inisialisasi Firebase
cred = credentials.Certificate(service_account_path)
firebase_admin.initialize_app(cred)

# Inisialisasi Firestore
db = firestore.client()

# Inisialisasi Cloud Storage
client = storage.Client.from_service_account_json(service_account_path)
bucket_name = "metpen-face-recognition.appspot.com"
bucket = client.get_bucket(bucket_name)

# Inisialisasi Kamera
camera = cv2.VideoCapture(0)

# Haar Cascade
cascade_path = os.path.join(script_dir, "haarcascade_frontalface_default.xml")
faceClassifier = cv2.CascadeClassifier(cascade_path)

# Input dan Pengecekan Data Gambar Wajah Mahasiswa
img_id = 0

while True:
    nim = input("NIM: ")
    nama = input("Nama: ")

    # Cek NIM di Firebase Cloud Storage
    blobs = bucket.list_blobs(prefix=nim + "/")
    nim_exists_in_storage = any(True for _ in blobs)

    # Cek NIM di Firestore
    mahasiswa_ref = db.collection("mahasiswa").document(nim)
    nim_exists_in_firestore = mahasiswa_ref.get().exists
    if nim_exists_in_storage or nim_exists_in_firestore:
        print("NIM mahasiswa sudah terdaftar.")
        continue

    # Cek nama di Firestore
    query = db.collection("mahasiswa").where("nama", "==", nama)
    nama_exists_in_firestore = len(query.get()) > 0
    if nama_exists_in_firestore:
        print("Nama mahasiswa sudah terdaftar.")
        continue
    break

# Jendela OpenCV
while True:
    _,img = camera.read()
    key=cv2.waitKey(1)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mendeteksi Wajah
    faces=faceClassifier.detectMultiScale(gray, 1.9, 5)

    # Membuat Bentuk Kotak di Sekitar Wajah
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_cropped=img[y:y+h,x:x+w]
        img_id+=1
        print(img_id)
        face=cv2.resize(face_cropped,(450,450))
        face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Path untuk Menyimpan Dataset ke Cloud Storage
        storage_path = f"{nim}/{nim}.{img_id}.jpg"

        # Menyimpan Dataset
        blob = bucket.blob(storage_path)
        _, img_encoded = cv2.imencode(".jpg", face)
        blob.upload_from_string(img_encoded.tobytes(), content_type="image/jpeg")

        # Menyimpan Data Mahasiswa ke Firestore
        if img_id >= 100:
            mahasiswa_ref.set(
                {
                    "nim": nim,
                    "nama": nama
                },
                merge=True,
            )
            break

        cv2.putText(img, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 
                2, (0,255,0), 2, cv2.LINE_AA)

    # Menampilkan Output
    cv2.imshow('img', img)

    if key==113 or img_id==100:
        break

from train import train_classifier
train_classifier(nim)

# Menutup Kamera dan Jendela OpenCV
camera.release()
cv2.destroyAllWindows()