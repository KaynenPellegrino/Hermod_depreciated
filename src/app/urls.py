# src/app/urls.py

from fastapi import APIRouter
from src.app.views import router as app_router
from src.app.client_portal.views import router as client_portal_router
# Import other routers as needed

router = APIRouter()

# Include routes from the app
router.include_router(app_router)

# Include routes from the client portal
router.include_router(
    client_portal_router,
    prefix="/client",
    tags=["Client Portal"],
)

# You can include more routers here
