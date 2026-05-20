# backend/routers/reports.py
#
# WHAT THIS FILE DOES:
#   Community flood reporting system.
#   Users can:
#     1. Submit a flood report from their location
#     2. View reports near their location
#     3. View all recent reports for Tamil Nadu
#
# ALL REPORTS ARE SAVED IN FIRESTORE (permanent database)
# SUBMITTING A REPORT REQUIRES LOGIN (Firebase token)
# READING REPORTS IS PUBLIC (no login needed)
#
# FIRESTORE STRUCTURE:
#   Collection: "flood_reports"
#   Document ID: auto-generated
#   Fields:
#     user_uid       → who submitted it
#     user_name      → their display name
#     lat, lon       → where they are
#     street         → street name
#     city           → city name
#     water_depth_cm → how deep the water is (they estimate)
#     severity       → "LOW" / "MEDIUM" / "HIGH" (they choose)
#     description    → their text description
#     image_url      → optional photo link
#     verified       → false (becomes true after admin review)
#     timestamp      → when they submitted
#     upvotes        → how many others confirmed this report

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
import math

from config import db
from utils.firebase_auth import verify_token, verify_token_optional

router = APIRouter()


# ─────────────────────────────────────────────────────────────
# DATA MODELS
# These define the shape of JSON Flutter sends and receives
# ─────────────────────────────────────────────────────────────

class ReportInput(BaseModel):
    """
    What Flutter sends when a user submits a flood report.
    All fields your friend needs to collect in the app.
    """
    latitude:      float = Field(..., ge=7.5,  le=14.0,
                                  description="User's latitude")
    longitude:     float = Field(..., ge=76.0, le=81.0,
                                  description="User's longitude")
    street:        str   = Field(..., min_length=2,
                                  description="Street name where flood is seen")
    city:          str   = Field(..., min_length=2,
                                  description="City name")
    water_depth_cm: float = Field(..., ge=0, le=500,
                                   description="Estimated water depth in cm")
    severity:      str   = Field(...,
                                  description="LOW, MEDIUM, or HIGH")
    description:   str   = Field("", max_length=500,
                                  description="Optional text description")
    image_url:     Optional[str] = Field(None,
                                          description="Optional photo URL")


class UpvoteInput(BaseModel):
    """What Flutter sends when a user confirms someone else's report."""
    report_id: str = Field(..., description="The Firestore document ID")


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def calculate_distance_km(lat1, lon1, lat2, lon2) -> float:
    """
    Calculates straight-line distance between two coordinates.
    Uses Haversine formula — accurate for short distances.
    Returns distance in kilometres.
    """
    R = 6371  # Earth radius in km
    phi1  = math.radians(lat1)
    phi2  = math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)

    a = (math.sin(dphi/2)**2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def validate_severity(severity: str) -> str:
    """Makes sure severity is one of the three valid values."""
    valid = ["LOW", "MEDIUM", "HIGH"]
    upper = severity.upper()
    if upper not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"severity must be one of: {valid}. Got: '{severity}'"
        )
    return upper


# ─────────────────────────────────────────────────────────────
# ENDPOINT 1: Submit a flood report
# Requires login (Firebase token in header)
# ─────────────────────────────────────────────────────────────

@router.post("/reports/submit")
def submit_report(
    report: ReportInput,
    user:   dict = Depends(verify_token)   # ← verifies Firebase token
):
    """
    USER SUBMITS A FLOOD REPORT FROM THEIR LOCATION.

    Requires Authorization header with Firebase token.

    Flutter sends:
    {
      "latitude": 13.0068,
      "longitude": 80.2206,
      "street": "Velachery Main Road",
      "city": "Chennai",
      "water_depth_cm": 35.0,
      "severity": "HIGH",
      "description": "Water is up to knee level near the signal",
      "image_url": null
    }

    Header:
      Authorization: Bearer <firebase_id_token>
    """

    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Database not connected. Check Firebase config."
        )

    # Validate severity value
    severity = validate_severity(report.severity)

    # Build the document to save in Firestore
    report_doc = {
        # Who submitted it (from Firebase token — can't be faked)
        "user_uid":   user["uid"],
        "user_name":  user["name"],
        "user_email": user["email"],

        # Where the flood is
        "lat":    report.latitude,
        "lon":    report.longitude,
        "street": report.street.strip(),
        "city":   report.city.strip(),

        # Flood details (user-reported)
        "water_depth_cm": report.water_depth_cm,
        "severity":       severity,
        "description":    report.description.strip(),
        "image_url":      report.image_url,

        # Metadata
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "verified":   False,   # admin can set to True after review
        "upvotes":    0,       # other users can confirm this report
        "upvoted_by": [],      # list of user UIDs who upvoted

        # Auto-set color for Flutter UI
        "color": {
            "LOW":    "#1D9E75",
            "MEDIUM": "#EF9F27",
            "HIGH":   "#E24B4A"
        }.get(severity, "#EF9F27")
    }

    # Save to Firestore — auto-generates a unique document ID
    doc_ref = db.collection("flood_reports").add(report_doc)

    # doc_ref is a tuple: (timestamp, DocumentReference)
    doc_id = doc_ref[1].id

    print(f"Report saved: {doc_id} by {user['name']} at {report.street}")

    return {
        "success":    True,
        "report_id":  doc_id,
        "message":    "Flood report submitted successfully",
        "saved_data": {
            "street":        report_doc["street"],
            "city":          report_doc["city"],
            "severity":      severity,
            "depth_cm":      report.water_depth_cm,
            "submitted_by":  user["name"],
            "at":            report_doc["timestamp"]
        }
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINT 2: Get reports near a location
# Public — no login required
# ─────────────────────────────────────────────────────────────

@router.get("/reports/nearby")
def get_nearby_reports(
    lat:           float,
    lon:           float,
    radius_km:     float = 10.0,   # default: show reports within 10km
    max_results:   int   = 20
):
    """
    RETURNS FLOOD REPORTS NEAR THE USER'S LOCATION.
    Public endpoint — no login needed.

    Flutter calls:
      GET /api/reports/nearby?lat=13.0068&lon=80.2206&radius_km=10

    Returns reports within radius_km kilometres of lat/lon,
    sorted by most recent first.
    """

    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    # Get recent reports from Firestore
    # We get more than needed and filter by distance in Python
    # (Firestore doesn't support geo-radius queries natively)
    try:
        docs = (
            db.collection("flood_reports")
            .order_by("timestamp", direction="DESCENDING")
            .limit(200)   # get last 200, then filter by distance
            .stream()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database query failed: {str(e)}"
        )

    nearby = []
    for doc in docs:
        data    = doc.to_dict()
        rep_lat = data.get("lat", 0)
        rep_lon = data.get("lon", 0)

        # Calculate distance from user to this report
        dist_km = calculate_distance_km(lat, lon, rep_lat, rep_lon)

        # Only include if within the requested radius
        if dist_km <= radius_km:
            nearby.append({
                "report_id":     doc.id,
                "street":        data.get("street", ""),
                "city":          data.get("city", ""),
                "lat":           rep_lat,
                "lon":           rep_lon,
                "severity":      data.get("severity", "MEDIUM"),
                "color":         data.get("color", "#EF9F27"),
                "water_depth_cm":data.get("water_depth_cm", 0),
                "description":   data.get("description", ""),
                "image_url":     data.get("image_url"),
                "user_name":     data.get("user_name", "Anonymous"),
                "verified":      data.get("verified", False),
                "upvotes":       data.get("upvotes", 0),
                "timestamp":     data.get("timestamp", ""),
                "distance_km":   round(dist_km, 2)
            })

        if len(nearby) >= max_results:
            break

    # Sort by distance (closest first)
    nearby.sort(key=lambda x: x["distance_km"])

    return {
        "reports":     nearby,
        "count":       len(nearby),
        "radius_km":   radius_km,
        "centre":      {"lat": lat, "lon": lon}
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINT 3: Get all recent reports for Tamil Nadu
# Public — no login needed
# ─────────────────────────────────────────────────────────────

@router.get("/reports/recent")
def get_recent_reports(limit: int = 50):
    """
    RETURNS THE MOST RECENT FLOOD REPORTS ACROSS ALL TAMIL NADU.
    Public endpoint — no login needed.
    Useful for the main map screen showing all community reports.

    Flutter calls:
      GET /api/reports/recent?limit=50
    """

    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    try:
        docs = (
            db.collection("flood_reports")
            .order_by("timestamp", direction="DESCENDING")
            .limit(limit)
            .stream()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database query failed: {str(e)}"
        )

    reports = []
    for doc in docs:
        data = doc.to_dict()
        reports.append({
            "report_id":      doc.id,
            "street":         data.get("street", ""),
            "city":           data.get("city", ""),
            "lat":            data.get("lat", 0),
            "lon":            data.get("lon", 0),
            "severity":       data.get("severity", "MEDIUM"),
            "color":          data.get("color", "#EF9F27"),
            "water_depth_cm": data.get("water_depth_cm", 0),
            "description":    data.get("description", ""),
            "image_url":      data.get("image_url"),
            "user_name":      data.get("user_name", "Anonymous"),
            "verified":       data.get("verified", False),
            "upvotes":        data.get("upvotes", 0),
            "timestamp":      data.get("timestamp", "")
        })

    return {
        "reports": reports,
        "count":   len(reports)
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINT 4: Upvote a report (confirm it is real)
# Requires login — prevents spam upvoting
# ─────────────────────────────────────────────────────────────

@router.post("/reports/upvote")
def upvote_report(
    data: UpvoteInput,
    user: dict = Depends(verify_token)
):
    """
    USER CONFIRMS SOMEONE ELSE'S FLOOD REPORT IS ACCURATE.
    Each user can upvote a report only once.
    More upvotes = more trustworthy report.

    Flutter sends:
    {
      "report_id": "abc123xyz"
    }

    Header:
      Authorization: Bearer <firebase_id_token>
    """

    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    # Get the report document from Firestore
    doc_ref = db.collection("flood_reports").document(data.report_id)
    doc     = doc_ref.get()

    if not doc.exists:
        raise HTTPException(
            status_code=404,
            detail=f"Report '{data.report_id}' not found"
        )

    doc_data    = doc.to_dict()
    upvoted_by  = doc_data.get("upvoted_by", [])

    # Check if this user already upvoted
    if user["uid"] in upvoted_by:
        return {
            "success": False,
            "message": "You have already confirmed this report",
            "upvotes": doc_data.get("upvotes", 0)
        }

    # Prevent upvoting your own report
    if doc_data.get("user_uid") == user["uid"]:
        return {
            "success": False,
            "message": "You cannot upvote your own report"
        }

    # Add this user's UID to upvoted_by and increment count
    new_upvotes = doc_data.get("upvotes", 0) + 1
    doc_ref.update({
        "upvotes":    new_upvotes,
        "upvoted_by": upvoted_by + [user["uid"]]
    })

    return {
        "success": True,
        "message": "Thank you for confirming this flood report",
        "upvotes": new_upvotes
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINT 5: Get a single report by ID
# Public — no login needed
# ─────────────────────────────────────────────────────────────

@router.get("/reports/{report_id}")
def get_report(report_id: str):
    """
    RETURNS ONE SPECIFIC REPORT BY ITS ID.

    Flutter calls:
      GET /api/reports/abc123xyz
    """

    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    doc = db.collection("flood_reports").document(report_id).get()

    if not doc.exists:
        raise HTTPException(
            status_code=404,
            detail=f"Report '{report_id}' not found"
        )

    data = doc.to_dict()
    return {
        "report_id":      doc.id,
        "street":         data.get("street", ""),
        "city":           data.get("city", ""),
        "lat":            data.get("lat", 0),
        "lon":            data.get("lon", 0),
        "severity":       data.get("severity", ""),
        "color":          data.get("color", ""),
        "water_depth_cm": data.get("water_depth_cm", 0),
        "description":    data.get("description", ""),
        "image_url":      data.get("image_url"),
        "user_name":      data.get("user_name", "Anonymous"),
        "verified":       data.get("verified", False),
        "upvotes":        data.get("upvotes", 0),
        "timestamp":      data.get("timestamp", "")
    }