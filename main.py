import os
import io
import cv2
from datetime import datetime
import numpy as np
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

# Inisialisasi Google Cloud Storage client
client = storage.Client.from_service_account_json(service_account_path)

cascadePath = os.path.join(script_dir, "haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0

# Cari mahasiswa berdasarkan NIM
bucket_name = "metpen-face-recognition.appspot.com"
bucket = client.get_bucket(bucket_name)
folder_prefix = "mahasiswa/"

blobs = bucket.list_blobs(prefix=folder_prefix)
daftar_nim = []

for blob in blobs:
    blob_name = blob.name
    if "/" in blob_name and blob_name.endswith(".jpg"):
        nim = blob_name.split("/")[1]
        nim = nim.split(".")[0]
        daftar_nim.append(nim)

daftar_nim=list(set(daftar_nim))
print(daftar_nim)

# Load model classifier
classifier_path = "model/classifier.xml"
blob = bucket.blob(classifier_path)

# Simpan file classifier.xml ke sistem file lokal
temp_classifier_path = os.path.join(script_dir, "temp_classifier.xml")
blob.download_to_filename(temp_classifier_path)

# Baca file classifier.xml menggunakan OpenCV
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(temp_classifier_path)

cam = cv2.VideoCapture(0)

while True:
    # Mendapatkan waktu saat ini
    now = datetime.now()

    # Format waktu ke dalam string
    formatted_time = now.strftime("%H:%M:%S - %d/%m/%Y")

    ret, img =cam.read()
    #img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.9,
        minNeighbors = 5
    )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        print(id)
        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 100):
            id = daftar_nim[id-1]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        # Mendapatkan nama dari dokument 'mahasiswa' dengan ID yang diperoleh
        id_split = id.split('_')[0]
        nama = ""
        doc_ref = db.collection('mahasiswa').document(id_split)
        doc = doc_ref.get()
        if doc.exists:
            nama = doc.to_dict().get('nama')

        cv2.putText(img, nama, (x+5,y-5), font, 1, (255,255,255), 2)

        if id!="Unknown":
            print("NIM: ", id_split)
            print("Nama: ", nama)
            print("Waktu: ", formatted_time)
            print("Confidence: ", confidence)

    cv2.imshow('camera',img) 
    k = cv2.waitKey(1)
    if k == 113:
        break

# Keluar dari program
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

# Hapus file sementara
os.remove(temp_classifier_path)