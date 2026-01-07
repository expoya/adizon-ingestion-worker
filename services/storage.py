"""
MinIO Storage Service for document management.

Handles file uploads, downloads, and management in MinIO object storage.
Uses boto3 with run_in_executor for async compatibility.

Supports both:
- Config-based initialization (multi-tenant)
- Legacy singleton mode (backwards compatibility)
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from core.config import MinioConfig, get_settings

# Thread pool for running blocking boto3 operations
_executor = ThreadPoolExecutor(max_workers=4)


class MinioService:
    """
    Service for interacting with MinIO S3-compatible storage.
    """

    def __init__(self, config: MinioConfig):
        """
        Initialize MinIO client with explicit configuration.

        Args:
            config: MinIO connection configuration
        """
        self.config = config
        self.client = boto3.client(
            "s3",
            endpoint_url=f"{'https' if config.secure else 'http'}://{config.endpoint}",
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        self.bucket = config.bucket_name

    async def _run_sync(self, func, *args, **kwargs):
        """Run a synchronous function in the thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, partial(func, *args, **kwargs)
        )

    async def download_file(self, object_name: str) -> bytes:
        """Download a file from MinIO storage."""
        response = await self._run_sync(
            self.client.get_object,
            Bucket=self.bucket,
            Key=object_name,
        )
        return response["Body"].read()

    async def file_exists(self, object_name: str) -> bool:
        """Check if a file exists in storage."""
        try:
            await self._run_sync(
                self.client.head_object,
                Bucket=self.bucket,
                Key=object_name,
            )
            return True
        except ClientError:
            return False


def create_minio_service(config: MinioConfig) -> MinioService:
    """Create a new MinIO service instance with the given config."""
    return MinioService(config)


# =============================================================================
# Legacy singleton support (for backwards compatibility)
# =============================================================================
_minio_service: MinioService | None = None


def get_minio_service() -> MinioService:
    """
    Get or create MinIO service singleton using legacy .env settings.

    DEPRECATED: Use create_minio_service(config) for multi-tenant support.
    """
    global _minio_service
    if _minio_service is None:
        settings = get_settings()
        legacy_config = MinioConfig(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket_name=settings.minio_bucket_name,
            secure=settings.minio_secure,
        )
        _minio_service = MinioService(legacy_config)
    return _minio_service
