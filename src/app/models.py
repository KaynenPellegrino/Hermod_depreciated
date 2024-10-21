# src/app/models.py

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    Table,
)
from sqlalchemy.orm import relationship, backref
from datetime import datetime
from src.utils.database import Base

# Association table for many-to-many relationship between User and Project
user_project_association = Table(
    'user_project_association',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('project_id', Integer, ForeignKey('projects.id')),
)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(120), unique=True, index=True, nullable=False)
    hashed_password = Column(String(128), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    projects = relationship(
        "Project",
        secondary=user_project_association,
        back_populates="users",
    )

    feedbacks = relationship("Feedback", back_populates="user")

    def verify_password(self, password: str) -> bool:
        import bcrypt

        return bcrypt.checkpw(password.encode('utf-8'), self.hashed_password.encode('utf-8'))

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    users = relationship(
        "User",
        secondary=user_project_association,
        back_populates="projects",
    )

    feedbacks = relationship("Feedback", back_populates="project")

    def __repr__(self):
        return f"<Project(id={self.id}, name='{self.name}')>"

class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    subject = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    rating = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user_id = Column(Integer, ForeignKey('users.id'))
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=True)

    user = relationship("User", back_populates="feedbacks")
    project = relationship("Project", back_populates="feedbacks")

    def __repr__(self):
        return f"<Feedback(id={self.id}, subject='{self.subject}', user_id={self.user_id})>"

class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text, nullable=True)

    users = relationship("UserRole", back_populates="role")

    def __repr__(self):
        return f"<Role(id={self.id}, name='{self.name}')>"

class UserRole(Base):
    __tablename__ = "user_roles"

    user_id = Column(Integer, ForeignKey('users.id'), primary_key=True)
    role_id = Column(Integer, ForeignKey('roles.id'), primary_key=True)
    assigned_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", backref=backref("user_roles", cascade="all, delete-orphan"))
    role = relationship("Role", back_populates="users")

    def __repr__(self):
        return f"<UserRole(user_id={self.user_id}, role_id={self.role_id})>"
