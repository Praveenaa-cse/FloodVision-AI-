# backend/utils/firebase_auth.py
#
# WHAT THIS DOES:
#   Verifies the Firebase ID Token that Flutter sends with every request.
#
# HOW TOKEN VERIFICATION WORKS:
#   1. User signs in with Google in Flutter
#   2. Firebase gives Flutter an ID Token (a long string like "eyJhbGci...")
#   3. Flutter puts this token in the request header:
#        Authorization: Bearer eyJhbGci...
#   4. Your backend receives the request
#   5. THIS FILE checks with Firebase: "Is this token real and not expired?"
#   6. Firebase confirms → you get the user's info (uid, email, name)
#   7. You trust the request and process it
#
# WHY THIS IS SECURE:
#   - Tokens expire after 1 hour → old stolen tokens don't work
#   - Firebase signs tokens cryptographically → can't be faked
#   - You never see the user's password → Firebase handles that

from fastapi import HTTPException, Header
from firebase_admin import auth
from typing import Optional


def verify_token(authorization: Optional[str] = Header(None)) -> dict:
    """
    Verifies the Firebase ID Token from the request header.

    HOW FLUTTER SENDS THE TOKEN:
      In every API call that needs login, Flutter adds a header:
        'Authorization': 'Bearer <the_id_token>'

    HOW TO USE THIS IN YOUR ENDPOINTS:
      from utils.firebase_auth import verify_token
      from fastapi import Depends

      @router.post("/submit-report")
      def submit_report(
          report_data: ReportInput,
          user: dict = Depends(verify_token)   ← this line verifies automatically
      ):
          user_id    = user["uid"]     # user's unique Firebase ID
          user_email = user["email"]   # user's email address
          user_name  = user["name"]    # user's display name

    RETURNS:
      dict with uid, email, name if token is valid

    RAISES:
      401 error if token is missing, expired, or fake
    """

    # Check if Authorization header exists
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail={
                "error":   "No token provided",
                "message": "Include 'Authorization: Bearer <token>' header",
                "fix":     "Flutter should add the Firebase ID token to every request"
            }
        )

    # The header format is "Bearer <token>"
    # We split it to get just the token part
    parts = authorization.split(" ")

    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail={
                "error":   "Wrong header format",
                "message": "Header must be: 'Authorization: Bearer <token>'",
                "received": authorization[:50]  # show first 50 chars for debug
            }
        )

    id_token = parts[1]

    # Verify the token with Firebase
    # Firebase checks: is this token real? is it expired? was it tampered with?
    try:
        decoded = auth.verify_id_token(id_token)

        # Return the user's information extracted from the token
        return {
            "uid":   decoded.get("uid", ""),           # unique user ID
            "email": decoded.get("email", ""),         # email address
            "name":  decoded.get("name", "Anonymous"), # display name
            "picture": decoded.get("picture", "")      # profile photo URL
        }

    except auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=401,
            detail={
                "error":   "Token expired",
                "message": "User must sign in again to get a fresh token",
                "fix":     "Flutter: call user.getIdToken(forceRefresh: true)"
            }
        )

    except auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=401,
            detail={
                "error":   "Invalid token",
                "message": "Token is malformed or was tampered with",
                "fix":     "Flutter: use FirebaseAuth.instance.currentUser?.getIdToken()"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail={
                "error":   "Token verification failed",
                "message": str(e)
            }
        )


def verify_token_optional(
        authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """
    Same as verify_token but does NOT raise an error if no token.
    Use this for endpoints that work for both logged-in and guest users.

    Returns user dict if token is valid, None if no token provided.
    """
    if not authorization:
        return None   # guest user — no error
    try:
        return verify_token(authorization)
    except HTTPException:
        return None   # invalid token — treat as guest, no error