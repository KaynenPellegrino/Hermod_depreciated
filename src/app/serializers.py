# src/app/serializers.py

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str = Field(..., min_length=6, max_length=128)

class UserRead(UserBase):
    id: int
    is_active: bool
    is_superuser: bool
    created_at: datetime

    class Config:
        orm_mode = True

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=6, max_length=128)

class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class ProjectRead(ProjectBase):
    id: int
    created_at: datetime
    users: List[UserRead] = []

    class Config:
        orm_mode = True

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class FeedbackBase(BaseModel):
    subject: str
    message: str
    rating: Optional[int] = Field(None, ge=1, le=5)

class FeedbackCreate(FeedbackBase):
    pass

class FeedbackRead(FeedbackBase):
    id: int
    created_at: datetime
    user_id: int
    project_id: Optional[int] = None

    class Config:
        orm_mode = True

class FeedbackUpdate(BaseModel):
    subject: Optional[str] = None
    message: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)
