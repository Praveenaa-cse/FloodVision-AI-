# backend/config.py
#
# WHAT THIS DOES:
#   - Loads all secret keys from .env
#   - Initializes Firebase app ONCE (used by both auth + firestore)
#   - Every other file imports from here
#
# WHY INITIALIZE FIREBASE HERE:
#   Firebase can only be initialized once in the entire Python process.
#   If two files both try to initialize it, you get an error.
#   So we do it here once, and every other file just imports the result.

import os
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────
OPENWEATHER_KEY  = os.getenv("OPENWEATHER_API_KEY", "")
FIREBASE_CRED    = os.getenv("FIREBASE_CRED",
                             "firebase-service-account.json")

if not OPENWEATHER_KEY:
    print("WARNING: OPENWEATHER_API_KEY not set in .env")

# ── Initialize Firebase (only once) ──────────────────────────
# This single initialization is shared by:
#   - firebase_auth.py  (verifying user tokens)
#   - notifications.py  (sending push alerts)
#   - reports.py        (saving to Firestore)

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CRED)
        firebase_admin.initialize_app(cred)
        print("Firebase initialized successfully")
    else:
        print("Firebase already initialized")

except FileNotFoundError:
    print(f"ERROR: Firebase credential file not found at '{FIREBASE_CRED}'")
    print("Make sure firebase-service-account.json is in the backend/ folder")

except Exception as e:
    print(f"ERROR: Firebase initialization failed: {e}")

# ── Firestore database client ─────────────────────────────────
# This is the object you use to read/write to Firestore
# Import this in any file that needs the database:
#   from config import db
try:
    db = firestore.client()
    print("Firestore connected successfully")
except Exception as e:
    db = None
    print(f"WARNING: Firestore connection failed: {e}")