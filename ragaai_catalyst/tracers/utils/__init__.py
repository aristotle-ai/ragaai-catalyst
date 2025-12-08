"""
Shared utilities for tracing
"""
import re
from urllib.parse import urlparse, urlunparse
from ragaai_catalyst.tracers.utils.logger import get_logger


def update_presigned_url(presigned_url, base_url):
    """
    Replaces the domain (and port, if applicable) of the presigned URL
    with that of the base URL only if the base URL contains 'localhost' or an IP address.
    """
    presigned_parts = urlparse(presigned_url)
    base_parts = urlparse(base_url)
    # Check if base_url contains localhost or an IP address
    if re.match(r"^(localhost|\d{1,3}(\.\d{1,3}){3})$", base_parts.hostname):
        new_netloc = base_parts.hostname  # Extract domain from base_url
        if base_parts.port:  # Add port if present in base_url
            new_netloc += f":{base_parts.port}"
        updated_parts = presigned_parts._replace(netloc=new_netloc)
        return urlunparse(updated_parts)
    return presigned_url