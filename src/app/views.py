# src/app/views.py

from fastapi import APIRouter, Request, Depends, status, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from src.app.models import User
from src.app.forms import LoginForm, RegisterForm
from src.utils.database import get_db
from src.utils.security import create_access_token, verify_password
from src.utils.decorators import login_required

router = APIRouter()
templates = Jinja2Templates(directory="src/app/templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login")
async def login_post(
    request: Request,
    form_data: LoginForm = Depends(LoginForm.as_form),
    db: Session = Depends(get_db),
):
    # Authenticate user
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not user.verify_password(form_data.password):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid username or password"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    # Create access token and set it in a cookie
    access_token = create_access_token(data={"sub": user.username})
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
    return response

@router.get("/register", response_class=HTMLResponse)
async def register_get(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@router.post("/register")
async def register_post(
    request: Request,
    form_data: RegisterForm = Depends(RegisterForm.as_form),
    db: Session = Depends(get_db),
):
    # Check if user already exists
    existing_user = db.query(User).filter(User.username == form_data.username).first()
    if existing_user:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Username already taken"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    # Hash the password and create a new user
    hashed_password = User.hash_password(form_data.password)
    new_user = User(
        username=form_data.username,
        email=form_data.email,
        hashed_password=hashed_password,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    # Redirect to login page
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

@router.get("/dashboard", response_class=HTMLResponse)
@login_required
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@router.get("/logout")
async def logout(request: Request):
    response = RedirectResponse(url="/")
    response.delete_cookie(key="access_token")
    return response
