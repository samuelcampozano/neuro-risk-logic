"""
JWT Authentication module for protecting API endpoints.
Implements Bearer token authentication for admin operations.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from app.config import settings
from loguru import logger

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer token scheme
security = HTTPBearer()


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token data extracted from JWT."""
    username: Optional[str] = None
    scopes: list[str] = []
    exp: Optional[datetime] = None


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time
        
    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    
    # Convert to timestamp (important: use timestamp() not just the datetime)
    to_encode.update({"exp": expire.timestamp()})
    
    try:
        encoded_jwt = jwt.encode(
            to_encode,
            settings.secret_key,
            algorithm=settings.jwt_algorithm
        )
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating access token: {str(e)}")
        raise


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenData:
    """
    Verify JWT token from Bearer authentication.
    
    Args:
        credentials: HTTP Bearer credentials
        
    Returns:
        TokenData: Decoded token information
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    token = credentials.credentials
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        
        username: str = payload.get("sub")
        scopes: list = payload.get("scopes", [])
        exp: float = payload.get("exp")
        
        if username is None:
            raise credentials_exception
        
        token_data = TokenData(
            username=username,
            scopes=scopes,
            exp=datetime.fromtimestamp(exp) if exp else None
        )
        
        return token_data
        
    except JWTError as e:
        logger.warning(f"JWT validation error: {str(e)}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Unexpected error in token verification: {str(e)}")
        raise credentials_exception


def verify_api_key(api_key: str) -> bool:
    """
    Verify if the provided API key is valid.
    
    Args:
        api_key: API key to verify
        
    Returns:
        bool: True if valid, False otherwise
    """
    return api_key == settings.api_key


def get_admin_token(expires_delta: Optional[timedelta] = None) -> Token:
    """
    Generate an admin token for protected operations.
    
    Args:
        expires_delta: Optional custom expiration time
        
    Returns:
        Token: JWT token with admin privileges
    """
    if not expires_delta:
        expires_delta = timedelta(minutes=settings.access_token_expire_minutes)
    
    access_token = create_access_token(
        data={
            "sub": "admin",
            "scopes": ["admin", "read", "write", "retrain"]
        },
        expires_delta=expires_delta
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=int(expires_delta.total_seconds())
    )


def hash_password(password: str) -> str:
    """
    Hash a password for storage.
    
    Args:
        password: Plain text password
        
    Returns:
        str: Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        bool: True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


# Dependency functions for protected routes
async def require_token(
    token_data: TokenData = Depends(verify_token)
) -> TokenData:
    """
    Dependency to require valid JWT token.
    
    Args:
        token_data: Decoded token data
        
    Returns:
        TokenData: Token data if valid
        
    Raises:
        HTTPException: If token is invalid
    """
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if token is expired
    if token_data.exp and token_data.exp < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return token_data


async def require_admin(
    token_data: TokenData = Depends(verify_token)
) -> TokenData:
    """
    Dependency to require admin privileges.
    
    Args:
        token_data: Decoded token data
        
    Returns:
        TokenData: Token data if user has admin scope
        
    Raises:
        HTTPException: If user doesn't have admin privileges
    """
    if "admin" not in token_data.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return token_data


async def require_write_permission(
    token_data: TokenData = Depends(verify_token)
) -> TokenData:
    """
    Dependency to require write permission.
    
    Args:
        token_data: Decoded token data
        
    Returns:
        TokenData: Token data if user has write scope
        
    Raises:
        HTTPException: If user doesn't have write permission
    """
    if "write" not in token_data.scopes and "admin" not in token_data.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Write permission required"
        )
    return token_data


async def require_retrain_permission(
    token_data: TokenData = Depends(verify_token)
) -> TokenData:
    """
    Dependency to require model retrain permission.
    
    Args:
        token_data: Decoded token data
        
    Returns:
        TokenData: Token data if user has retrain scope
        
    Raises:
        HTTPException: If user doesn't have retrain permission
    """
    if "retrain" not in token_data.scopes and "admin" not in token_data.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Model retrain permission required"
        )
    return token_data


# Optional: API Key authentication for simple use cases
from fastapi import Header


async def get_api_key_header(
    x_api_key: str = Header(..., alias="X-API-Key")
) -> str:
    """
    Extract and validate API key from header.
    
    Args:
        x_api_key: API key from X-API-Key header
        
    Returns:
        str: API key if valid
        
    Raises:
        HTTPException: If API key is invalid
    """
    if not verify_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    return x_api_key