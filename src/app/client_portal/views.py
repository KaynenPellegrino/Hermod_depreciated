# src/app/client_portal/views.py

from typing import List, Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from src.app.models import Project, Feedback, Contact, User
from src.app.client_portal.forms import (
    ProjectSubmissionForm,
    FeedbackForm,
    ContactForm,
)
from src.app.client_portal.serializers import (
    ProjectResponse,
    FeedbackResponse,
    ContactResponse,
    ProjectListResponse,
    FeedbackListResponse,
    ContactListResponse,
    APIResponse,
)
from src.utils.validation_utils import sizeof_fmt
from src.utils.helpers import sizeof_fmt
import logging

# Initialize logger
logger = logging.getLogger(__name__)

def submit_project_view(
    form: ProjectSubmissionForm,
    db: Session
) -> APIResponse:
    """
    Handles the submission of a new project by a client.
    """
    logger.info("Submitting a new project.")
    try:
        # Process attachments
        attachments = None
        if form.attachments and form.attachment_sizes:
            attachments = []
            for filename, size in zip(form.attachments, form.attachment_sizes):
                if size > 10 * 1024 * 1024:  # 10MB limit
                    readable_size = sizeof_fmt(size)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Attachment '{filename}' size {readable_size} exceeds the 10MB limit."
                    )
                attachments.append(filename)  # Or store URLs/paths as needed

        # Create a new Project instance
        project = Project(
            name=form.project_name,
            description=form.project_description,
            client_email=form.client_email,
            client_phone=form.client_phone,
            website_url=str(form.website_url) if form.website_url else None,
            start_date=form.start_date,
            end_date=form.end_date,
            budget=form.budget,
            attachments=','.join(attachments) if attachments else None
        )

        # Add to the database
        db.add(project)
        db.commit()
        db.refresh(project)

        # Serialize the response
        project_response = ProjectResponse.from_orm(project)
        logger.info(f"Project '{project.name}' submitted successfully with ID {project.id}.")
        return APIResponse(
            success=True,
            message="Project submitted successfully.",
            data=project_response
        )
    except HTTPException as he:
        logger.error(f"HTTPException during project submission: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during project submission: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while submitting the project."
        )

def submit_feedback_view(
    form: FeedbackForm,
    db: Session
) -> APIResponse:
    """
    Handles the submission of feedback for a project by a client.
    """
    logger.info("Submitting feedback for a project.")
    try:
        # Verify that the project exists
        project = db.query(Project).filter(Project.id == form.project_id).first()
        if not project:
            logger.warning(f"Project with ID {form.project_id} not found.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found."
            )

        # Optionally, verify that the user is associated with the project
        # Assuming authentication is handled and user_id is available
        # Here, we skip user association for simplicity

        # Create a new Feedback instance
        feedback = Feedback(
            subject=form.subject,
            message=form.message,
            rating=form.rating,
            user_id=form.user_id,  # Assuming form includes user_id
            project_id=form.project_id
        )

        # Add to the database
        db.add(feedback)
        db.commit()
        db.refresh(feedback)

        # Serialize the response
        feedback_response = FeedbackResponse.from_orm(feedback)
        logger.info(f"Feedback ID {feedback.id} submitted successfully for Project ID {project.id}.")
        return APIResponse(
            success=True,
            message="Feedback submitted successfully.",
            data=feedback_response
        )
    except HTTPException as he:
        logger.error(f"HTTPException during feedback submission: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during feedback submission: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while submitting the feedback."
        )

def submit_contact_view(
    form: ContactForm,
    db: Session
) -> APIResponse:
    """
    Handles the submission of a contact form by a user.
    """
    logger.info("Submitting a contact form.")
    try:
        # Create a new Contact instance
        contact = Contact(
            name=form.name,
            email=form.email,
            subject=form.subject,
            message=form.message,
            subscribe=form.subscribe
        )

        # Add to the database
        db.add(contact)
        db.commit()
        db.refresh(contact)

        # Serialize the response
        contact_response = ContactResponse.from_orm(contact)
        logger.info(f"Contact form submitted successfully by '{contact.name}' with ID {contact.id}.")
        return APIResponse(
            success=True,
            message="Contact form submitted successfully.",
            data=contact_response
        )
    except Exception as e:
        logger.error(f"Unexpected error during contact form submission: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while submitting the contact form."
        )

def list_projects_view(
    page: int = 1,
    size: int = 10,
    db: Session
) -> ProjectListResponse:
    """
    Retrieves a paginated list of all projects.
    """
    logger.info(f"Listing projects - Page: {page}, Size: {size}")
    if page < 1 or size < 1:
        logger.warning("Invalid pagination parameters.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page and size must be positive integers."
        )
    try:
        offset = (page - 1) * size
        projects = db.query(Project).offset(offset).limit(size).all()
        total = db.query(Project).count()
        project_responses = [ProjectResponse.from_orm(project) for project in projects]
        logger.info(f"Retrieved {len(project_responses)} projects out of {total}.")
        return ProjectListResponse(
            projects=project_responses,
            total=total,
            page=page,
            size=size
        )
    except Exception as e:
        logger.error(f"Unexpected error while listing projects: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the projects."
        )

def list_feedbacks_view(
    page: int = 1,
    size: int = 10,
    db: Session
) -> FeedbackListResponse:
    """
    Retrieves a paginated list of all feedback entries.
    """
    logger.info(f"Listing feedbacks - Page: {page}, Size: {size}")
    if page < 1 or size < 1:
        logger.warning("Invalid pagination parameters for feedbacks.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page and size must be positive integers."
        )
    try:
        offset = (page - 1) * size
        feedbacks = db.query(Feedback).offset(offset).limit(size).all()
        total = db.query(Feedback).count()
        feedback_responses = [FeedbackResponse.from_orm(feedback) for feedback in feedbacks]
        logger.info(f"Retrieved {len(feedback_responses)} feedbacks out of {total}.")
        return FeedbackListResponse(
            feedbacks=feedback_responses,
            total=total,
            page=page,
            size=size
        )
    except Exception as e:
        logger.error(f"Unexpected error while listing feedbacks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the feedback entries."
        )

def list_contacts_view(
    page: int = 1,
    size: int = 10,
    db: Session
) -> ContactListResponse:
    """
    Retrieves a paginated list of all contact form submissions.
    """
    logger.info(f"Listing contacts - Page: {page}, Size: {size}")
    if page < 1 or size < 1:
        logger.warning("Invalid pagination parameters for contacts.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page and size must be positive integers."
        )
    try:
        offset = (page - 1) * size
        contacts = db.query(Contact).offset(offset).limit(size).all()
        total = db.query(Contact).count()
        contact_responses = [ContactResponse.from_orm(contact) for contact in contacts]
        logger.info(f"Retrieved {len(contact_responses)} contacts out of {total}.")
        return ContactListResponse(
            contacts=contact_responses,
            total=total,
            page=page,
            size=size
        )
    except Exception as e:
        logger.error(f"Unexpected error while listing contacts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the contact entries."
        )
