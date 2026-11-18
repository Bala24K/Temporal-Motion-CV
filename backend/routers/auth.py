from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..config import settings
from ..db import User, UnsupportedRequestUser
from ..deps import current_user, get_db
from ..security import create_token, hash_password, login_limiter, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])
GOALS = {"build_muscle", "get_stronger", "improve_fitness", "lose_weight", "consistency", "movement_quality"}


class Credentials(BaseModel):
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=8, max_length=128)
    name: str = Field(default="", max_length=120)


def _user_out(u: User) -> dict:
    return {"id": u.id, "email": u.email, "name": u.name, "is_admin": u.is_admin, "goal": u.goal,
            "weekly_target": u.weekly_target}


def _session_response(u: User, response: Response) -> dict:
    tok = create_token(u.id, settings.secret_key, settings.token_ttl_seconds)
    response.set_cookie("session", tok, httponly=True, samesite="lax", secure=settings.cookie_secure,
                        max_age=settings.token_ttl_seconds)
    return {"token": tok, "user": _user_out(u)}


@router.get("/config")
def auth_config():
    from ..state import state
    return {"local": True, "oidc": ({"issuer": state.oidc.issuer, "audience": state.oidc.audience} if state.oidc else None)}


@router.post("/register", status_code=201)
def register(body: Credentials, response: Response, db: Session = Depends(get_db)):
    email = body.email.lower()
    if db.query(User).filter_by(email=email).first():
        raise HTTPException(409, "Email already registered")
    u = User(email=email, password_hash=hash_password(body.password), name=body.name,
             is_admin=email in settings.admin_emails)
    db.add(u)
    db.commit()
    return _session_response(u, response)


@router.post("/login")
def login(body: Credentials, request: Request, response: Response, db: Session = Depends(get_db)):
    ip = request.client.host if request.client else "?"
    if not login_limiter.allow(f"{ip}|{body.email.lower()}"):
        raise HTTPException(429, "Too many attempts. Wait a minute and try again.")
    u = db.query(User).filter_by(email=body.email.lower()).first()
    if not u or not verify_password(body.password, u.password_hash):
        raise HTTPException(401, "Invalid email or password")
    return _session_response(u, response)


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie("session")
    return {"ok": True}


@router.get("/me")
def me(u: User = Depends(current_user)):
    return _user_out(u)


class Prefs(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    goal: str | None = None
    weekly_target: int | None = Field(default=None, ge=1, le=14)


@router.patch("/me")
def update_me(body: Prefs, u: User = Depends(current_user), db: Session = Depends(get_db)):
    if body.goal is not None:
        if body.goal not in GOALS:
            raise HTTPException(422, f"goal must be one of {sorted(GOALS)}")
        u.goal = body.goal
    if body.name is not None:
        u.name = body.name
    if body.weekly_target is not None:
        u.weekly_target = body.weekly_target
    db.merge(u)
    db.commit()
    return _user_out(u)


@router.delete("/me", status_code=204)
def delete_me(response: Response, u: User = Depends(current_user), db: Session = Depends(get_db)):
    """Privacy: removes the account and every derived record (no raw video is ever stored)."""
    db.query(UnsupportedRequestUser).filter_by(user_id=u.id).delete()
    db.delete(u)
    db.commit()
    response.delete_cookie("session")
