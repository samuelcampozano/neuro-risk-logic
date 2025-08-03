"""
Authentication routes for JWT token management.
"""

from datetime import timedelta
from fastapi import APIRouter, HTTPException, status, Depends
from app.schemas.request import LoginRequest
from app.schemas.response import TokenResponse
from app.auth import get_admin_token, verify_api_key, verify_token, TokenData
from app.config import settings
from loguru import logger

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login with API key",
    description="Exchange API key for JWT token",
)
async def login(request: LoginRequest) -> TokenResponse:
    """
    Authenticate with API key and receive JWT token.

    Args:
        request: Login request with API key

    Returns:
        JWT token for authenticated requests
    """
    if not verify_api_key(request.api_key):
        logger.warning(f"Failed login attempt with invalid API key")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    # Generate token with appropriate scopes
    # In a real system, you'd look up user permissions from database
    token = get_admin_token(expires_delta=timedelta(minutes=settings.access_token_expire_minutes))

    logger.info("Successful login with API key")

    return TokenResponse(
        access_token=token.access_token,
        token_type=token.token_type,
        expires_in=token.expires_in,
        scopes=["read", "write", "admin", "retrain"],
    )


@router.post("/verify", summary="Verify JWT token", description="Check if a JWT token is valid")
async def verify_jwt_token(token_data: TokenData = Depends(verify_token)) -> dict:
    """
    Verify that a JWT token is valid.

    Args:
        token_data: Decoded token data (from dependency)

    Returns:
        Token validation result
    """
    return {
        "valid": True,
        "username": token_data.username,
        "scopes": token_data.scopes,
        "expires": token_data.exp.isoformat() if token_data.exp else None,
    }


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh JWT token",
    description="Get a new JWT token using existing valid token",
)
async def refresh_token(token_data: TokenData = Depends(verify_token)) -> TokenResponse:
    """
    Refresh JWT token before expiration.

    Args:
        token_data: Current valid token data

    Returns:
        New JWT token
    """
    # Generate new token with same permissions
    from app.auth import create_access_token

    new_token = create_access_token(
        data={"sub": token_data.username, "scopes": token_data.scopes},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
    )

    logger.info(f"Token refreshed for user: {token_data.username}")

    return TokenResponse(
        access_token=new_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
        scopes=token_data.scopes,
    )
