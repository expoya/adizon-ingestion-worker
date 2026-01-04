"""
MinIO Storage Service for document management.

Handles file uploads, downloads, and management in MinIO object storage.
Uses boto3 with run_in_executor for async compatibility.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from core.config import get_settings

settings = get_settings()

# Thread pool for running blocking boto3 operations
_executor = ThreadPoolExecutor(max_workers=4)


class MinioService:
    """
    Service for interacting with MinIO S3-compatible storage.
    """

    def __init__(self):
        """Initialize MinIO client."""
        self.client = boto3.client(
            "s3",
            endpoint_url=f"{'https' if settings.minio_secure else 'http'}://{settings.minio_endpoint}",
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        self.bucket = settings.minio_bucket_name

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


# Singleton instance
_minio_service: MinioService | None = None


def get_minio_service() -> MinioService:
    """Get or create MinIO service singleton."""
    global _minio_service
    if _minio_service is None:
        _minio_service = MinioService()
    return _minio_service
