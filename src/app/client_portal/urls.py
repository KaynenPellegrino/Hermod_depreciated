# src/app/client_portal/urls.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.app.client_portal.forms import (
    ProjectSubmissionForm,
    FeedbackForm,
    ContactForm,
)
from src.app.client_portal.views import (
    submit_project_view,
    submit_feedback_view,
    submit_contact_view,
    list_projects_view,
    list_feedbacks_view,
    list_contacts_view,
)
from src.utils.database import get_db
from src.utils.decorators import authenticate
from src.app.client_portal.serializers import APIResponse, ProjectListResponse, FeedbackListResponse, ContactListResponse

router = APIRouter(
    prefix="/client-portal",
    tags=["Client Portal"],
    dependencies=[Depends(authenticate)],  # Apply authentication to all routes
    responses={404: {"description": "Not Found"}},
)

@router.post(
    "/submit-project",
    response_model=APIResponse,
    summary="Submit a New Project",
    description="Allows clients to submit a new project with detailed information."
)
async def submit_project(
    form: ProjectSubmissionForm,
    db: Session = Depends(get_db)
):
    return submit_project_view(form, db)

@router.post(
    "/submit-feedback",
    response_model=APIResponse,
    summary="Submit Feedback for a Project",
    description="Allows clients to submit feedback for a completed project."
)
async def submit_feedback(
    form: FeedbackForm,
    db: Session = Depends(get_db)
):
    return submit_feedback_view(form, db)

@router.post(
    "/contact",
    response_model=APIResponse,
    summary="Submit a Contact Form",
    description="Allows users to submit a contact form for inquiries or support."
)
async def submit_contact(
    form: ContactForm,
    db: Session = Depends(get_db)
):
    return submit_contact_view(form, db)

@router.get(
    "/projects",
    response_model=ProjectListResponse,
    summary="List All Projects",
    description="Retrieves a paginated list of all projects submitted by clients."
)
async def list_projects(
    page: int = 1,
    size: int = 10,
    db: Session = Depends(get_db)
):
    return list_projects_view(page, size, db)

@router.get(
    "/feedbacks",
    response_model=FeedbackListResponse,
    summary="List All Feedbacks",
    description="Retrieves a paginated list of all feedback entries submitted by clients."
)
async def list_feedbacks(
    page: int = 1,
    size: int = 10,
    db: Session = Depends(get_db)
):
    return list_feedbacks_view(page, size, db)

@router.get(
    "/contacts",
    response_model=ContactListResponse,
    summary="List All Contacts",
    description="Retrieves a paginated list of all contact form submissions."
)
async def list_contacts(
    page: int = 1,
    size: int = 10,
    db: Session = Depends(get_db)
):
    return list_contacts_view(page, size, db)
