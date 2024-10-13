# data_management/models/data_models.py

from datetime import datetime
from typing import Optional, List, Any, Dict

from pydantic import BaseModel, Field, field_validator


class BaseDataModel(BaseModel):
    """
    Base model for all data sources.
    """
    id: Optional[int] = Field(None, description="Unique identifier for the record")
    name: str = Field(..., description="Name associated with the record")
    value: float = Field(..., description="Numeric value representing the record")
    timestamp: Optional[datetime] = Field(None, description="Timestamp of the record creation or update")

    @field_validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.utcnow()


class GitHubRepoDataModel(BaseDataModel):
    """
    Data model for GitHub repository data.
    """
    full_name: str = Field(..., description="Full name of the repository (e.g., 'octocat/Hello-World')")
    description: Optional[str] = Field(None, description="Description of the repository")
    language: Optional[str] = Field(None, description="Primary language used in the repository")
    stargazers_count: int = Field(..., description="Number of stars the repository has")
    forks_count: int = Field(..., description="Number of forks of the repository")
    open_issues_count: int = Field(..., description="Number of open issues in the repository")
    watchers_count: int = Field(..., description="Number of watchers of the repository")
    topics: List[str] = Field(default_factory=list, description="List of topics associated with the repository")
    license: Optional[str] = Field(None, description="License of the repository")
    url: str = Field(..., description="URL of the repository")

    @field_validator('url')
    def validate_url(cls, v):
        assert v.startswith('http'), 'URL must start with http or https'
        return v


class APIDataModel(BaseDataModel):
    """
    Data model for generic API data.
    """
    data: Any = Field(..., description="Data fetched from the API")
    api_source: str = Field(..., description="Name or identifier of the API source")
    response_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata from the API response")

    @field_validator('api_source')
    def validate_api_source(cls, v):
        assert isinstance(v, str) and len(v) > 0, 'API source must be a non-empty string'
        return v


class FileDataModel(BaseDataModel):
    """
    Data model for data ingested from files.
    """
    file_path: str = Field(..., description="Path to the source file")
    file_format: str = Field(..., description="Format of the source file (e.g., 'csv', 'json')")
    size_bytes: Optional[int] = Field(None, description="Size of the file in bytes")
    checksum: Optional[str] = Field(None, description="Checksum of the file for integrity verification")
    read_options: Optional[Dict[str, Any]] = Field(default_factory=dict,
                                                   description="Additional read options used during ingestion")

    @field_validator('file_path')
    def validate_file_path(cls, v):
        assert isinstance(v, str) and len(v) > 0, 'File path must be a non-empty string'
        return v

    @field_validator('file_format')
    def validate_file_format(cls, v):
        supported_formats = ['csv', 'json', 'xlsx', 'xls', 'parquet']
        assert v.lower() in supported_formats, f"Unsupported file format: {v}. Supported formats are: {supported_formats}"
        return v.lower()


class TwitterDataModel(BaseDataModel):
    """
    Data model for Twitter API data.
    """
    tweet_id: str = Field(..., description="Unique identifier for the tweet")
    text: str = Field(..., description="Content of the tweet")
    user_id: str = Field(..., description="Unique identifier for the user")
    created_at: datetime = Field(..., description="Timestamp when the tweet was created")
    retweet_count: int = Field(..., description="Number of retweets")
    favorite_count: int = Field(..., description="Number of likes")
    hashtags: List[str] = Field(default_factory=list, description="List of hashtags used in the tweet")
    mentions: List[str] = Field(default_factory=list, description="List of user mentions in the tweet")
    language: Optional[str] = Field(None, description="Language of the tweet")

    @field_validator('tweet_id', 'user_id')
    def validate_ids(cls, v):
        assert isinstance(v, str) and len(v) > 0, 'ID fields must be non-empty strings'
        return v

    @field_validator('text')
    def validate_text(cls, v):
        assert len(v) > 0, 'Tweet text cannot be empty'
        return v

# Additional specific models can be defined similarly for other data sources like Reddit, Facebook, etc.
