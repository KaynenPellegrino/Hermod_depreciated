# src/app/client_portal/serializers.py

from typing import List, Optional, Any
from pydantic import BaseModel, EmailStr, HttpUrl, Field
from datetime import datetime
from src.app.models import Project, Feedback, Contact
from src.utils.helpers import sizeof_fmt

# Pydantic models for serialization/deserialization


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: str
    client_email: EmailStr
    client_phone: Optional[str] = None
    website_url: Optional[HttpUrl] = None
    start_date: datetime
    end_date: datetime
    budget: float
    attachments: Optional[List[str]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class FeedbackResponse(BaseModel):
    id: int
    project_id: int
    client_email: EmailStr
    rating: int = Field(..., ge=1, le=5)
    comments: Optional[str] = None
    would_recommend: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class ContactResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    subject: str
    message: str
    subscribe: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class ProjectListResponse(BaseModel):
    projects: List[ProjectResponse]
    total: int
    page: int
    size: int

    class Config:
        orm_mode = True


class FeedbackListResponse(BaseModel):
    feedbacks: List[FeedbackResponse]
    total: int
    page: int
    size: int

    class Config:
        orm_mode = True


class ContactListResponse(BaseModel):
    contacts: List[ContactResponse]
    total: int
    page: int
    size: int

    class Config:
        orm_mode = True


class APIResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True
