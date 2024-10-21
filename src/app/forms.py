# src/app/forms.py

from typing import Optional
from pydantic import BaseModel, EmailStr, Field, validator
from fastapi import Form

class LoginForm(BaseModel):
    """
    Form for user login.
    """

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)

    @classmethod
    def as_form(
        cls,
        username: str = Form(...),
        password: str = Form(...),
    ):
        return cls(username=username, password=password)


class RegisterForm(BaseModel):
    """
    Form for user registration.
    """

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr = Field(...)
    password: str = Field(..., min_length=6, max_length=128)
    confirm_password: str = Field(..., min_length=6, max_length=128)

    @validator("confirm_password")
    def passwords_match(cls, v, values, **kwargs):
        if "password" in values and v != values["password"]:
            raise ValueError("Passwords do not match")
        return v

    @classmethod
    def as_form(
        cls,
        username: str = Form(...),
        email: EmailStr = Form(...),
        password: str = Form(...),
        confirm_password: str = Form(...),
    ):
        return cls(
            username=username,
            email=email,
            password=password,
            confirm_password=confirm_password,
        )


class FeedbackForm(BaseModel):
    """
    Form for collecting user feedback.
    """

    subject: str = Field(..., max_length=100)
    message: str = Field(..., max_length=1000)
    rating: Optional[int] = Field(None, ge=1, le=5)

    @classmethod
    def as_form(
        cls,
        subject: str = Form(...),
        message: str = Form(...),
        rating: Optional[int] = Form(None),
    ):
        return cls(subject=subject, message=message, rating=rating)
