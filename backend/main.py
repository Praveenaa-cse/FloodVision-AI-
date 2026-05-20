# backend/main.py
# THE MAIN FILE — run this to start FloodVision AI
#
# HOW TO RUN:
#   cd backend
#   uvicorn main:app --reload --port 8000
#
# THEN OPEN:
#   http://localhost:8000/docs   ← interactive test page
#   http://localhost:8000/       ← health check

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import predict, streets

# Create the FastAPI application
app = FastAPI(
    title       = "FloodVision AI",
    description = "Street-level flood prediction for Tamil Nadu",
    version     = "1.0.0"
)

# Allow the React frontend to call this API
# Without this, browsers block cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # in production: change to your domain
    allow_methods  = ["*"],
    allow_headers  = ["*"]
)

# Register the routers (connect endpoints to app)
app.include_router(predict.router, prefix="/api")
app.include_router(streets.router, prefix="/api")


# Health check — visit this to confirm the server is running
@app.get("/")
def health():
    return {
        "app":     "FloodVision AI",
        "status":  "running",
        "version": "1.0.0",
        "docs":    "http://localhost:8000/docs",
        "coverage":"All Tamil Nadu + Coastal Areas"
    }