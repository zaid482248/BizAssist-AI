"""
File validation logic.
Yeh module uploaded files ko validate karta hai (extension, size).
"""

from pathlib import Path

from bizassist_ai.core.config import settings


class ValidationError(Exception):
    """Jab file valid na ho."""
    pass


# Allowed MIME types (extra safety layer)
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
}


def validate_extension(filename: str) -> str:
    """
    Check karta hai ke file extension allowed hai.
    
    Returns:
        Lowercase extension (e.g., ".pdf")
    
    Raises:
        ValidationError: Agar extension allowed nahi.
    """
    if not filename:
        raise ValidationError("Filename khali hai.")

    extension = Path(filename).suffix.lower()

    if not extension:
        raise ValidationError("File mein extension nahi hai.")

    if extension not in settings.allowed_extensions:
        raise ValidationError(
            f"Extension '{extension}' allowed nahi. "
            f"Allowed: {', '.join(settings.allowed_extensions)}"
        )

    return extension


def validate_size(size_bytes: int) -> None:
    """
    Check karta hai ke file size limit ke andar hai.
    
    Raises:
        ValidationError: Agar size zyada hai.
    """
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    if size_bytes > max_bytes:
        size_mb = size_bytes / (1024 * 1024)
        raise ValidationError(
            f"File bohat bari hai ({size_mb:.2f} MB). "
            f"Maximum allowed: {settings.max_upload_size_mb} MB"
        )

    if size_bytes == 0:
        raise ValidationError("File khali hai.")


def safe_filename(filename: str) -> str:
    """
    Filename ko safe banata hai (path traversal rokne ke liye).
    
    Example: "../../etc/passwd" -> "passwd"
    """
    # Sirf filename part lein, folder part nahi
    name = Path(filename).name

    # Sirf alphanumeric, dash, underscore, dot allow karein
    safe = "".join(c for c in name if c.isalnum() or c in "-_. ")

    if not safe:
        raise ValidationError("Filename mein koi valid character nahi.")

    return safe
