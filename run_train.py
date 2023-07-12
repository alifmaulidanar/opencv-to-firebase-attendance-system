import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud import storage

script_dir = os.path.dirname(os.path.abspath(__file__))

# Path ke file service account Firebase Admin SDK
service_account_path = os.path.join(script_dir, "service_account.json")

cred = credentials.Certificate(service_account_path)
firebase_admin.initialize_app(cred)

# Inisialisasi Firestore
db = firestore.client()

# Inisialisasi Cloud Storage
client = storage.Client.from_service_account_json(service_account_path)
bucket_name = "metpen-face-recognition.appspot.com"
bucket = client.get_bucket(bucket_name)
folder_prefix = f"mahasiswa/"
mahasiswa = bucket.list_blobs(prefix=folder_prefix)

from train import train_classifier
train_classifier(mahasiswa)