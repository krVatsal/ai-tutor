import os
import jwt
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends, Header
from google.auth.transport import requests
from google.oauth2 import id_token
from jose import JWTError, jwt as jose_jwt
from sqlalchemy.orm import Session
from database import get_db, UserProfile

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

class GoogleAuth:
    def __init__(self):
        self.client_id = GOOGLE_CLIENT_ID

    def verify_google_token(self, token: str) -> Dict[str, Any]:
        """Verify Google ID token and return user info"""
        try:
            # Verify the token with Google
            idinfo = id_token.verify_oauth2_token(
                token, requests.Request(), self.client_id
            )

            # Verify the issuer
            if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
                raise ValueError('Wrong issuer.')

            return idinfo
        except ValueError as e:
            raise HTTPException(status_code=401, detail=f"Invalid Google token: {str(e)}")

    def create_access_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        to_encode = {
            "sub": user_data["sub"],
            "email": user_data["email"],
            "given_name": user_data.get("given_name", ""),
            "family_name": user_data.get("family_name", ""),
            "picture": user_data.get("picture", ""),
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
            "iat": datetime.utcnow(),
        }
        return jose_jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

google_auth = GoogleAuth()

def verify_token(authorization: str = Header(None)) -> Dict[str, Any]:
    """Verify JWT token and return user data"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    try:
        # Extract token from "Bearer <token>"
        token = authorization.split(" ")[1] if authorization.startswith("Bearer ") else authorization
        
        # Decode and verify JWT token
        payload = jose_jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        # Check if token is expired
        exp = payload.get('exp')
        if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
            raise HTTPException(status_code=401, detail="Token has expired")
        
        return payload
        
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {str(e)}")

def create_or_update_user_profile(db: Session, user_data: Dict[str, Any]) -> UserProfile:
    """Create or update user profile from Google user data"""
    user_id = user_data.get("sub")
    email = user_data.get("email")
    first_name = user_data.get("given_name")
    last_name = user_data.get("family_name")
    profile_image_url = user_data.get("picture")
    
    # Check if user exists
    user_profile = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    
    if user_profile:
        # Update existing user
        user_profile.email = email
        user_profile.first_name = first_name
        user_profile.last_name = last_name
        user_profile.profile_image_url = profile_image_url
        user_profile.updated_at = datetime.utcnow()
        user_profile.last_login = datetime.utcnow()
    else:
        # Create new user
        user_profile = UserProfile(
            id=user_id,
            email=email,
            first_name=first_name,
            last_name=last_name,
            profile_image_url=profile_image_url,
            last_login=datetime.utcnow()
        )
        db.add(user_profile)
    
    db.commit()
    db.refresh(user_profile)
    return user_profile

def get_current_user(
    user_data: Dict[str, Any] = Depends(verify_token),
    db: Session = Depends(get_db)
) -> UserProfile:
    """Get current authenticated user"""
    return create_or_update_user_profile(db, user_data)
