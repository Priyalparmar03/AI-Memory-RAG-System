from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import json
import mimetypes


@dataclass
class Metadata:
    """Compact document/file metadata model."""

    filename: str
    path: str = ""
    extension: str = ""
    mime_type: str = ""
    size: int = 0
    checksum: str = ""
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs) -> None:
        """Update standard or custom metadata."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extra[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
            default=str,
        )


class MetadataExtractor:
    """Extract metadata from local files."""

    @staticmethod
    def checksum(path: Path, algorithm: str = "sha256") -> str:
        """Calculate file checksum."""
        try:
            hasher = hashlib.new(algorithm)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported hash algorithm: {algorithm}"
            ) from exc

        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(8192),
                b"",
            ):
                hasher.update(chunk)

        return hasher.hexdigest()

    @classmethod
    def extract(
        cls,
        file_path: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Metadata:
        """Extract basic metadata from a file."""

        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        stat = path.stat()
        mime_type, _ = mimetypes.guess_type(path.name)

        return Metadata(
            filename=path.name,
            path=str(path.resolve()),
            extension=path.suffix.lower(),
            mime_type=mime_type or "application/octet-stream",
            size=stat.st_size,
            checksum=cls.checksum(path),
            created_at=datetime.fromtimestamp(
                stat.st_ctime
            ).isoformat(),
            modified_at=datetime.fromtimestamp(
                stat.st_mtime
            ).isoformat(),
            extra=extra or {},
        )

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> Metadata:
        """Create Metadata from a dictionary."""

        fields = {
            "filename",
            "path",
            "extension",
            "mime_type",
            "size",
            "checksum",
            "created_at",
            "modified_at",
            "extra",
        }

        values = {
            key: value
            for key, value in data.items()
            if key in fields
        }

        return Metadata(**values)
