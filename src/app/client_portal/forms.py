# src/app/client_portal/forms.py

import re
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr, HttpUrl, validator
from src.utils.validation_utils import validate_password_strength, validate_email_custom, validate_url_custom
from datetime import datetime


class ProjectSubmissionForm(BaseModel):
    project_name: str = Field(..., min_length=3, max_length=100, description="Name of the project")
    project_description: str = Field(..., min_length=10, max_length=1000, description="Detailed description of the project")
    client_email: EmailStr = Field(..., description="Client's email address")
    client_phone: Optional[str] = Field(None, description="Client's phone number")
    website_url: Optional[HttpUrl] = Field(None, description="Client's website URL")
    start_date: datetime = Field(..., description="Project start date")
    end_date: datetime = Field(..., description="Project end date")
    budget: float = Field(..., gt=0, description="Budget allocated for the project")
    attachments: Optional[List[str]] = Field(None, description="List of attachment filenames or URLs")

    @validator('client_phone')
    def validate_phone_number(cls, v):
        """
        Validates the phone number format.
        Accepts formats like +1234567890, 123-456-7890, (123) 456-7890, etc.
        """
        if v:
            phone_pattern = re.compile(r'^(\+\d{1,3}[- ]?)?\d{10}$')
            if not phone_pattern.match(v):
                raise ValueError('Invalid phone number format')
        return v

    @validator('end_date')
    def validate_dates(cls, v, values, **kwargs):
        """
        Ensures that the end_date is after the start_date.
        """
        start_date = values.get('start_date')
        if start_date and v < start_date:
            raise ValueError('End date must be after start date')
        return v


class FeedbackForm(BaseModel):
    project_id: int = Field(..., description="ID of the project being reviewed")
    client_email: EmailStr = Field(..., description="Client's email address")
    rating: int = Field(..., ge=1, le=5, description="Rating out of 5")
    comments: Optional[str] = Field(None, max_length=1000, description="Additional comments or feedback")
    would_recommend: bool = Field(..., description="Would the client recommend our services?")

    @validator('comments')
    def validate_comments(cls, v):
        """
        Validates that comments do not contain prohibited content.
        """
        prohibited_words = ['badword1', 'badword2']  # Example prohibited words
        if v:
            for word in prohibited_words:
                if word in v.lower():
                    raise ValueError(f"Comments contain prohibited word: {word}")
        return v


# Example of another form: ContactForm
class ContactForm(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Name of the person")
    email: EmailStr = Field(..., description="Email address")
    subject: str = Field(..., min_length=5, max_length=150, description="Subject of the message")
    message: str = Field(..., min_length=10, max_length=2000, description="Detailed message")
    subscribe: Optional[bool] = Field(False, description="Subscribe to newsletter")

    @validator('message')
    def validate_message(cls, v):
        """
        Validates that the message does not contain any prohibited content.
        """
        prohibited_phrases = ['spam', 'advertisement']  # Example prohibited phrases
        if v:
            for phrase in prohibited_phrases:
                if phrase in v.lower():
                    raise ValueError(f"Message contains prohibited phrase: {phrase}")
        return v
