from fastapi import APIRouter
from pydantic import BaseModel

class LoginRequest(BaseModel):
    username: str
    password: str

router = APIRouter()

@router.post("/login")
async def login(creds: LoginRequest):
    # Mock Auth
    if creds.username == "admin" and creds.password == "admin":
        return {"token": "admin_token_123", "role": "admin", "user_id": "admin"}
    return {"token": "user_token_456", "role": "user", "user_id": creds.username}
