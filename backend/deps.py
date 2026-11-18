from typing import Iterator, Optional

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .config import settings
from .db import User
from .security import decode_token
from .state import state


def get_db() -> Iterator[Session]:
    db = state.SessionLocal()
    try:
        yield db
    finally:
        db.close()


def token_from_request(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:]
    return request.cookies.get("session")


def user_from_token(db: Session, token: Optional[str]) -> Optional[User]:
    """Local HS256 session token first, then (if configured) an OIDC access token from the identity provider.
    OIDC users are matched/created by *verified* e-mail only."""
    if not token:
        return None
    uid = decode_token(token, settings.secret_key)
    if uid:
        return db.get(User, uid)
    if state.oidc:
        claims = state.oidc.verify(token)
        if claims and claims.get("email") and claims.get("email_verified") is True:
            email = str(claims["email"]).lower()
            u = db.query(User).filter_by(email=email).first()
            if not u:
                u = User(email=email, password_hash="!oidc", name=str(claims.get("name", ""))[:120],
                         is_admin=email in settings.admin_emails)
                db.add(u)
                db.commit()
            return u
    return None


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    user = user_from_token(db, token_from_request(request))
    if not user:
        raise HTTPException(401, "Not authenticated")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(403, "Admin only")
    return user
