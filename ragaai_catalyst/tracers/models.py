"""
Domain models and value objects for tracer components.

This module contains immutable value objects that represent domain concepts
used across exporter, pipeline, and uploader layers.
Follows Domain-Driven Design principles.
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, NewType, Optional

TraceId = NewType('TraceId', str)
HashId = NewType('HashId', str)


@dataclass(frozen=True)
class TraceFile:
    """
    Immutable value object representing a trace file.

    Attributes:
        path: Filesystem path to the trace file
        trace_id: Unique identifier for the trace
        created_at: Timestamp when file was created
        size_bytes: File size in bytes (0 if not yet created)
    """
    path: Path
    trace_id: TraceId
    created_at: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0

    def exists(self) -> bool:
        return self.path.exists()

    def get_size(self) -> int:
        if self.exists():
            return self.path.stat().st_size
        return 0

    def cleanup(self) -> None:
        if self.exists():
            self.path.unlink()


@dataclass(frozen=True)
class CodeArchive:
    """
    Immutable value object representing a source code archive.

    Attributes:
        path: Filesystem path to the zip archive
        hash_id: Unique hash identifying the code snapshot
        file_count: Number of files in the archive
    """
    path: Path
    hash_id: HashId
    file_count: int = 0

    def exists(self) -> bool:
        return self.path.exists()

    def cleanup(self) -> None:
        if self.exists():
            self.path.unlink()


@dataclass(frozen=True)
class TraceMetadata:
    """
    Metadata associated with a trace.

    Attributes:
        system_info: System information dictionary
        resources: Resource usage information
        user_metadata: User-provided metadata
        project_name: Name of the project
        dataset_name: Name of the dataset
        tracer_type: Type of tracer (e.g., 'langchain', 'openai')
    """
    system_info: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    user_metadata: Dict[str, Any] = field(default_factory=dict)
    project_name: str = ""
    dataset_name: str = ""
    tracer_type: str = ""


@dataclass(frozen=True)
class TraceData:
    """
    Complete trace data in RagaAI format.

    Attributes:
        trace_id: Unique trace identifier
        data: Raw trace data dictionary
        metadata: Associated metadata
        workflow: Workflow interactions
    """
    trace_id: TraceId
    data: Dict[str, Any]
    metadata: TraceMetadata
    workflow: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedTrace:
    """
    Trace fully prepared and ready for upload.

    This is the final artifact produced by the trace preparation pipeline,
    containing all necessary files and metadata for upload.

    Attributes:
        trace_file: Serialized trace file
        code_archive: Source code archive
        trace_data: Complete trace data
    """
    trace_file: TraceFile
    code_archive: CodeArchive
    trace_data: TraceData

    def validate(self) -> bool:
        """
        Validate that all required files exist.

        Returns:
            True if trace is ready for upload, False otherwise
        """
        return self.trace_file.exists() and self.code_archive.exists()

    def cleanup(self) -> None:
        """Clean up all temporary files."""
        self.trace_file.cleanup()
        self.code_archive.cleanup()


@dataclass
class ProcessingContext:
    """
    Context information passed through the processing pipeline.

    Attributes:
        trace_id: Unique trace identifier
        project_name: Name of the project
        dataset_name: Name of the dataset
        base_url: API base URL
        tracer_type: Type of tracer
        timeout: Upload timeout in seconds
        files_to_zip: List of source files to include
        user_details: User-provided details
        user_context: Optional user context string
        user_gt: Optional ground truth string
        external_id: Optional external identifier
        code_hash: Hash of source code archive
        code_zip_path: Path to code archive
    """
    trace_id: TraceId
    project_name: str
    dataset_name: str
    base_url: str
    tracer_type: str
    timeout: int = 120
    files_to_zip: list = field(default_factory=list)
    user_details: Optional[Dict] = None
    user_context: Optional[str] = None
    user_gt: Optional[str] = None
    external_id: Optional[str] = None
    code_hash: Optional[HashId] = None
    code_zip_path: Optional[Path] = None


@dataclass
class UploadTask:
    """
    Internal upload task data structure.

    This represents the internal format used by the upload system.
    Converted from UploadRequest for compatibility with existing upload logic.

    Attributes:
        filepath: Path to the trace JSON file
        hash_id: Hash ID of code archive (empty string if none)
        zip_path: Path to code archive zip file (empty string if none)
        project_name: Name of the project
        dataset_name: Name of the dataset
        user_details: User-provided details
        base_url: API base URL
        tracer_type: Type of tracer
        timeout: Upload timeout in seconds
    """
    filepath: str
    hash_id: str
    zip_path: str
    project_name: str
    dataset_name: str
    user_details: Dict[str, Any]
    base_url: str
    tracer_type: str
    timeout: int = 120


@dataclass(frozen=True)
class UploadRequest:
    """
    Type-safe upload request - replaces fragile dict contract between layers.

    This value object encapsulates all data needed for trace upload,
    providing compile-time safety and clear contracts between exporter and uploader.

    Shared by both exporter and uploader layers, eliminating dictionary coupling.

    Attributes:
        trace_file: Path to the trace JSON file
        code_archive: Source code archive (optional for non-agentic)
        project_name: Name of the project
        dataset_name: Name of the dataset
        base_url: API base URL
        tracer_type: Type of tracer
        user_details: User-provided details
        timeout: Upload timeout in seconds
    """
    trace_file: Path
    code_archive: Optional[CodeArchive]
    project_name: str
    dataset_name: str
    base_url: str
    tracer_type: str
    user_details: Dict[str, Any]
    timeout: int = 120

    @classmethod
    def from_prepared_trace(
        cls,
        prepared_trace: 'PreparedTrace',
        context: ProcessingContext
    ) -> 'UploadRequest':
        """
        Create UploadRequest from PreparedTrace and ProcessingContext.

        Args:
            prepared_trace: Fully prepared trace with files
            context: Processing context with metadata

        Returns:
            Type-safe UploadRequest
        """
        return cls(
            trace_file=prepared_trace.trace_file.path,
            code_archive=prepared_trace.code_archive,
            project_name=context.project_name,
            dataset_name=context.dataset_name,
            base_url=context.base_url,
            tracer_type=context.tracer_type,
            user_details=context.user_details or {},
            timeout=context.timeout
        )

    def to_upload_task(self) -> UploadTask:
        """
        Convert UploadRequest to internal UploadTask format.

        Returns:
            UploadTask compatible with existing upload logic
        """
        return UploadTask(
            filepath=str(self.trace_file),
            hash_id=self.code_archive.hash_id if self.code_archive else '',
            zip_path=str(self.code_archive.path) if self.code_archive else '',
            project_name=self.project_name,
            dataset_name=self.dataset_name,
            user_details=self.user_details,
            base_url=self.base_url,
            tracer_type=self.tracer_type,
            timeout=self.timeout
        )

