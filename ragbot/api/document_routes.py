"""
Document processing API endpoints
"""

import os
import uuid
from datetime import datetime
from pathlib import Path

from flask import Blueprint, abort, current_app, request, send_file
from werkzeug.utils import secure_filename

from ragbot.config import Config
from ragbot.db.database_service import DatabaseService
from ragbot.ingestion.document_service import DocumentService
from ragbot.utils.response_helpers import error_response, success_response

document_bp = Blueprint("documents", __name__)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


@document_bp.route("/upload", methods=["POST"])
def upload_document():
    """Upload a document file for processing"""
    try:
        if "file" not in request.files:
            return error_response("No file part in the request", 400)

        file = request.files["file"]
        if file.filename == "":
            return error_response("No file selected", 400)

        if not allowed_file(file.filename):
            return error_response("File type not allowed", 400)

        # Ensure upload folder exists
        upload_folder = Path(Config.UPLOAD_FOLDER)
        upload_folder.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        original_filename = secure_filename(file.filename)
        file_extension = Path(original_filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = upload_folder / unique_filename

        # Save file
        file.save(str(file_path))

        # Get file size
        file_size = file_path.stat().st_size

        # Determine mime type
        if file_extension.lower() == ".pdf":
            mime_type = "application/pdf"
        elif file_extension.lower() in [".md", ".markdown"]:
            mime_type = "text/markdown"
        else:
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

        # Create database record immediately
        document = DatabaseService.create_document(
            file_name=unique_filename,
            original_file_name=original_filename,
            file_path=str(file_path),
            file_size=file_size,
            mime_type=mime_type,
            metadata={"upload_method": "web_interface"},
        )

        return success_response(
            {
                "message": "File uploaded successfully",
                "file_id": document.id,
                "filename": unique_filename,
                "original_filename": original_filename,
                "file_path": str(file_path),
                "file_size": file_size,
                "document_id": document.id,
            }
        )

    except Exception as e:
        current_app.logger.error(f"Error uploading file: {str(e)}")
        return error_response(f"Upload failed: {str(e)}", 500)


@document_bp.route("/process", methods=["POST"])
def process_documents():
    """Process uploaded documents and create embeddings"""
    try:
        data = request.get_json()

        if not data:
            return error_response("No JSON data provided", 400)

        # Get processing parameters
        input_path = data.get("input_path", str(Config.UPLOAD_FOLDER))
        output_dir = data.get("output_dir", str(Config.PROCESSED_FOLDER))
        files = data.get("files", [])  # Optional: specific files to process
        extract_patterns = data.get(
            "extract_patterns", False
        )  # Optional: extract patterns from headings

        # Create service instance within the request context to ensure proper app context
        from ragbot.ingestion.document_service import DocumentService

        service = DocumentService()

        # Process documents
        result = service.process_documents(
            input_path=input_path,
            output_dir=output_dir,
            files=files,
            extract_patterns=extract_patterns,
        )

        if result["errors"]:
            return error_response(
                {
                    "message": "Some documents failed to process",
                    "success": result["success"],
                    "errors": result["errors"],
                },
                207,
            )  # Multi-Status

        return success_response(
            {
                "message": "All documents processed successfully",
                "processed_files": result["success"],
                "pattern_extraction": result.get("pattern_extraction"),
            }
        )

    except Exception as e:
        current_app.logger.error(f"Error processing documents: {str(e)}")
        return error_response(f"Processing failed: {str(e)}", 500)


@document_bp.route("/status/<job_id>", methods=["GET"])
def get_processing_status(job_id):
    """Get the status of a document processing job"""
    try:
        # This would need to be implemented with a job tracking system
        # For now, return a placeholder response
        return success_response(
            {
                "job_id": job_id,
                "status": "completed",
                "message": "Processing completed successfully",
            }
        )

    except Exception as e:
        return error_response(f"Failed to get job status: {str(e)}", 500)


@document_bp.route("/files", methods=["GET"])
def list_files():
    """List available files from database"""
    try:
        # Get documents from database with chunk counts (more efficient)
        documents = DatabaseService.get_all_documents_with_chunk_count()

        # Organize by status - all documents go to processed array for frontend compatibility
        files_info = {
            "processed": documents,  # Frontend expects all files here
            "raw": [],  # Keep for backward compatibility
        }

        return success_response(
            {"directories": files_info, "files": documents}  # Legacy support
        )

    except Exception as e:
        return error_response(f"Failed to list files: {str(e)}", 500)


@document_bp.route("/delete/<path:filename>", methods=["DELETE"])
def delete_file(filename):
    """Delete a file from the system"""
    try:
        file_path = Path(filename)

        # Security check - only allow deletion from specific directories
        allowed_dirs = [
            Config.UPLOAD_FOLDER,
            Config.PROCESSED_FOLDER,
            Config.TEMP_FOLDER,
        ]
        if not any(
            str(file_path).startswith(str(allowed_dir)) for allowed_dir in allowed_dirs
        ):
            return error_response("Cannot delete file from this directory", 403)

        if not file_path.exists():
            return error_response("File not found", 404)

        file_path.unlink()

        return success_response({"message": f"File {filename} deleted successfully"})

    except Exception as e:
        return error_response(f"Failed to delete file: {str(e)}", 500)


@document_bp.route("/delete/document/<int:document_id>", methods=["DELETE"])
def delete_document(document_id):
    """Delete a document and all its associated data from the database and filesystem"""
    try:
        from ragbot.db.database_service import DatabaseService

        # Get document details
        document = DatabaseService.get_document_by_id(document_id)
        if not document:
            return error_response("Document not found", 404)

        # Delete associated chunks and embeddings first
        chunks_deleted = DatabaseService.delete_document_chunks(document_id)

        # Delete the document record
        document_deleted = DatabaseService.delete_document(document_id)

        # Delete physical files if they exist
        files_deleted = []
        if document.file_path and Path(document.file_path).exists():
            try:
                Path(document.file_path).unlink()
                files_deleted.append("source_file")
            except Exception as e:
                current_app.logger.warning(f"Could not delete source file: {e}")

        # Try to delete processed files
        processed_path = Path(Config.PROCESSED_FOLDER) / f"{document.file_name}.jsonl"
        if processed_path.exists():
            try:
                processed_path.unlink()
                files_deleted.append("processed_file")
            except Exception as e:
                current_app.logger.warning(f"Could not delete processed file: {e}")

        return success_response(
            {
                "message": f"Document {document.original_file_name} deleted successfully",
                "document_id": document_id,
                "chunks_deleted": chunks_deleted,
                "document_deleted": document_deleted,
                "files_deleted": files_deleted,
            }
        )

    except Exception as e:
        current_app.logger.error(f"Error deleting document: {str(e)}")
        return error_response(f"Failed to delete document: {str(e)}", 500)


@document_bp.route("/delete/raw", methods=["DELETE"])
def delete_raw_files():
    """Delete all raw uploaded files and their database records"""
    try:
        from ragbot.db.database_service import DatabaseService

        # Get all raw documents (not processed)
        raw_documents = DatabaseService.get_documents_by_status("uploaded")

        if not raw_documents:
            return success_response(
                {"message": "No raw files found to delete", "deleted_count": 0}
            )

        deleted_count = 0
        errors = []

        for doc in raw_documents:
            try:
                # Delete physical file
                if doc.file_path and Path(doc.file_path).exists():
                    Path(doc.file_path).unlink()

                # Delete database record
                DatabaseService.delete_document(doc.id)
                deleted_count += 1

            except Exception as e:
                errors.append(f"Failed to delete {doc.original_file_name}: {str(e)}")

        if errors:
            return error_response(
                {
                    "message": f"Some files failed to delete. {deleted_count} deleted successfully.",
                    "deleted_count": deleted_count,
                    "errors": errors,
                },
                207,
            )  # Multi-Status

        return success_response(
            {
                "message": "All raw files deleted successfully",
                "deleted_count": deleted_count,
            }
        )

    except Exception as e:
        current_app.logger.error(f"Error deleting raw files: {str(e)}")
        return error_response(f"Failed to delete raw files: {str(e)}", 500)


@document_bp.route("/delete/processed", methods=["DELETE"])
def delete_processed_files():
    """Delete all processed files, embeddings, and their database records"""
    try:
        from ragbot.db.database_service import DatabaseService

        # Get all processed documents
        processed_documents = DatabaseService.get_documents_by_status("completed")

        if not processed_documents:
            return success_response(
                {"message": "No processed files found to delete", "deleted_count": 0}
            )

        deleted_count = 0
        chunks_deleted = 0
        errors = []

        for doc in processed_documents:
            try:
                # Delete chunks and embeddings
                chunks_deleted += DatabaseService.delete_document_chunks(doc.id)

                # Delete physical files
                if doc.file_path and Path(doc.file_path).exists():
                    Path(doc.file_path).unlink()

                # Delete processed JSONL file
                processed_path = (
                    Path(Config.PROCESSED_FOLDER) / f"{doc.file_name}.jsonl"
                )
                if processed_path.exists():
                    processed_path.unlink()

                # Delete database record
                DatabaseService.delete_document(doc.id)
                deleted_count += 1

            except Exception as e:
                errors.append(f"Failed to delete {doc.original_file_name}: {str(e)}")

        if errors:
            return error_response(
                {
                    "message": f"Some processed files failed to delete. {deleted_count} deleted successfully.",
                    "deleted_count": deleted_count,
                    "chunks_deleted": chunks_deleted,
                    "errors": errors,
                },
                207,
            )  # Multi-Status

        return success_response(
            {
                "message": "All processed files deleted successfully",
                "deleted_count": deleted_count,
                "chunks_deleted": chunks_deleted,
            }
        )

    except Exception as e:
        current_app.logger.error(f"Error deleting processed files: {str(e)}")
        return error_response(f"Failed to delete processed files: {str(e)}", 500)


@document_bp.route("/delete/embeddings", methods=["DELETE"])
def delete_embeddings():
    """Delete all embeddings/chunks from the database while keeping documents"""
    try:
        from ragbot.db.database_service import DatabaseService

        # Delete all chunks
        chunks_deleted = DatabaseService.delete_all_chunks()

        # Update all documents to 'uploaded' status (remove 'completed' status)
        documents_updated = DatabaseService.reset_document_processing_status()

        return success_response(
            {
                "message": "All embeddings deleted successfully",
                "chunks_deleted": chunks_deleted,
                "documents_updated": documents_updated,
            }
        )

    except Exception as e:
        current_app.logger.error(f"Error deleting embeddings: {str(e)}")
        return error_response(f"Failed to delete embeddings: {str(e)}", 500)


@document_bp.route("/delete/all", methods=["DELETE"])
def delete_all_data():
    """Delete all data: files, database records, and embeddings"""
    try:
        from ragbot.db.database_service import DatabaseService

        # Get confirmation from request
        data = request.get_json() or {}
        confirm = data.get("confirm", False)

        if not confirm:
            return error_response(
                "Please confirm deletion by setting confirm=true in request body", 400
            )

        # Delete all chunks first
        chunks_deleted = DatabaseService.delete_all_chunks()

        # Delete all documents
        documents_deleted = DatabaseService.delete_all_documents()

        # Delete all model patterns
        patterns_deleted = DatabaseService.delete_all_model_patterns()

        # Delete physical files from upload folder
        files_deleted = 0
        if Config.UPLOAD_FOLDER.exists():
            for file_path in Config.UPLOAD_FOLDER.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        files_deleted += 1
                    except Exception as e:
                        current_app.logger.warning(
                            f"Could not delete file {file_path}: {e}"
                        )

        # Delete processed files
        processed_deleted = 0
        if Config.PROCESSED_FOLDER.exists():
            for file_path in Config.PROCESSED_FOLDER.iterdir():
                if file_path.is_file() and file_path.suffix == ".jsonl":
                    try:
                        file_path.unlink()
                        processed_deleted += 1
                    except Exception as e:
                        current_app.logger.warning(
                            f"Could not delete processed file {file_path}: {e}"
                        )

        return success_response(
            {
                "message": "All data deleted successfully",
                "chunks_deleted": chunks_deleted,
                "documents_deleted": documents_deleted,
                "patterns_deleted": patterns_deleted,
                "files_deleted": files_deleted,
                "processed_files_deleted": processed_deleted,
            }
        )

    except Exception as e:
        current_app.logger.error(f"Error deleting all data: {str(e)}")
        return error_response(f"Failed to delete all data: {str(e)}", 500)


@document_bp.route("/delete/embeddings/<int:document_id>", methods=["DELETE"])
def delete_document_embeddings(document_id):
    """Delete embeddings for a specific document while keeping the document"""
    try:
        from ragbot.db.database_service import DatabaseService

        # Check if document exists
        document = DatabaseService.get_document_by_id(document_id)
        if not document:
            return error_response("Document not found", 404)

        # Delete chunks for this document
        chunks_deleted = DatabaseService.delete_document_chunks(document_id)

        # Update document status to 'uploaded' (remove 'completed' status)
        DatabaseService.update_document_status(document_id, "uploaded")

        return success_response(
            {
                "message": f"Embeddings for {document.original_file_name} deleted successfully",
                "document_id": document_id,
                "chunks_deleted": chunks_deleted,
            }
        )

    except Exception as e:
        current_app.logger.error(f"Error deleting document embeddings: {str(e)}")
        return error_response(f"Failed to delete document embeddings: {str(e)}", 500)


@document_bp.route("/view/<int:document_id>", methods=["GET"])
def view_document(document_id):
    """View a document file by ID in browser"""
    try:
        # Get document from database
        document = DatabaseService.get_document_by_id(document_id)

        if not document:
            return abort(404, description="Document not found")

        # Get the file path
        file_path = Path(document.file_path)
        current_app.logger.info("ℹ️ File path: %s", file_path)
        current_dir = os.environ.get("CURRENT_DIR")
        current_app.logger.info("ℹ️ Current directory: %s", current_dir)
        full_file_path = os.path.join(current_dir, file_path)
        current_app.logger.info("ℹ️ Full file path: %s", full_file_path)
        # Check if file exists
        if not file_path.exists():
            current_app.logger.error(f"File not found: {file_path}")
            return abort(404, description="File not found on server")

        # Determine the MIME type based on file extension
        file_extension = file_path.suffix.lower()
        mimetype = get_mimetype_for_extension(file_extension)

        # Send the file for viewing in browser
        return send_file(
            full_file_path,
            mimetype=mimetype,
            as_attachment=False,
            download_name=document.original_file_name or document.file_name,
        )

    except Exception as e:
        current_app.logger.error(f"Error serving document {document_id}: {str(e)}")
        return abort(500, description=f"Error serving document: {str(e)}")


@document_bp.route("/download/<int:document_id>", methods=["GET"])
def download_document(document_id):
    """Download a document file by ID"""
    try:
        # Get document from database
        document = DatabaseService.get_document_by_id(document_id)

        if not document:
            return abort(404, description="Document not found")

        # Get the file path
        file_path = Path(document.file_path)

        # Check if file exists
        if not file_path.exists():
            current_app.logger.error(f"File not found: {file_path}")
            return abort(404, description="File not found on server")

        # Determine the MIME type based on file extension
        file_extension = file_path.suffix.lower()
        mimetype = get_mimetype_for_extension(file_extension)

        # Send the file as download
        return send_file(
            file_path,
            mimetype=mimetype,
            as_attachment=True,
            download_name=document.original_file_name or document.file_name,
        )

    except Exception as e:
        current_app.logger.error(f"Error downloading document {document_id}: {str(e)}")
        return abort(500, description=f"Error downloading document: {str(e)}")


@document_bp.route("/info/<int:document_id>", methods=["GET"])
def get_document_info(document_id):
    """Get document information and metadata"""
    try:
        # Get document from database
        document = DatabaseService.get_document_by_id(document_id)

        if not document:
            return error_response("Document not found", 404)

        # Get file stats if file exists
        file_path = Path(document.file_path)
        file_stats = None

        if file_path.exists():
            stat = file_path.stat()
            file_stats = {
                "size": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": file_path.suffix.lower(),
                "mimetype": get_mimetype_for_extension(file_path.suffix.lower()),
            }

        return success_response(
            {
                "document": {
                    "id": document.id,
                    "original_file_name": document.original_file_name,
                    "file_name": document.file_name,
                    "file_path": document.file_path,
                    "file_size": document.file_size,
                    "upload_date": (
                        document.upload_date.isoformat()
                        if document.upload_date
                        else None
                    ),
                    "processing_date": (
                        document.processing_date.isoformat()
                        if document.processing_date
                        else None
                    ),
                    "status": document.status,
                    "chunks_count": len(document.chunks) if document.chunks else 0,
                    "file_stats": file_stats,
                }
            }
        )

    except Exception as e:
        current_app.logger.error(f"Error getting document info {document_id}: {str(e)}")
        return error_response(f"Error getting document info: {str(e)}", 500)


def get_mimetype_for_extension(extension):
    """Get MIME type for file extension"""
    mimetypes = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".txt": "text/plain",
        ".rtf": "application/rtf",
        ".odt": "application/vnd.oasis.opendocument.text",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".ppt": "application/vnd.ms-powerpoint",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".svg": "image/svg+xml",
    }

    return mimetypes.get(extension, "application/octet-stream")


@document_bp.route("/upload-config", methods=["GET"])
def get_upload_config():
    """Get upload configuration including passcode requirement status"""
    try:
        # In a real application, you might want to read this from config or environment
        upload_config = {
            "passcode_required": True,
            "max_file_size_mb": (
                Config.MAX_CONTENT_LENGTH // (1024 * 1024)
                if hasattr(Config, "MAX_CONTENT_LENGTH")
                else 100
            ),
            "allowed_extensions": (
                list(Config.ALLOWED_EXTENSIONS)
                if hasattr(Config, "ALLOWED_EXTENSIONS")
                else [".pdf", ".docx", ".md", ".markdown"]
            ),
            "max_files_per_upload": 10,
        }

        return success_response({"config": upload_config})

    except Exception as e:
        current_app.logger.error(f"Error getting upload config: {str(e)}")
        return error_response(f"Error getting upload config: {str(e)}", 500)


@document_bp.route("/validate-passcode", methods=["POST"])
def validate_upload_passcode():
    """Validate upload passcode"""
    try:
        data = request.get_json()
        if not data:
            return error_response("No JSON data provided", 400)

        passcode = data.get("passcode", "").strip()
        if not passcode:
            return error_response("Passcode is required", 400)

        # In a real application, you should:
        # 1. Store the passcode hash in environment variables or secure config
        # 2. Use proper password hashing (bcrypt, etc.)
        # 3. Implement rate limiting to prevent brute force attacks
        # 4. Log authentication attempts for security monitoring

        # For demonstration, using environment variable or default
        import os

        correct_passcode = os.getenv("UPLOAD_PASSCODE", "upload123")

        if passcode == correct_passcode:
            return success_response(
                {"valid": True, "message": "Passcode validated successfully"}
            )
        else:
            # Log failed attempts (in production, implement proper security logging)
            current_app.logger.warning(
                f"Failed upload passcode attempt from IP: {request.remote_addr}"
            )
            return success_response({"valid": False, "message": "Invalid passcode"})

    except Exception as e:
        current_app.logger.error(f"Error validating passcode: {str(e)}")
        return error_response(f"Error validating passcode: {str(e)}", 500)


@document_bp.route("/extract-patterns", methods=["POST"])
def extract_patterns():
    """Extract patterns from documents in input directory"""
    try:
        data = request.get_json() or {}
        input_path = data.get("input_path", str(Config.DATA_DIR / "uploads"))
        files = data.get("files")  # Optional list of specific files

        # Validate input path exists
        if not Path(input_path).exists():
            return error_response(f"Input path does not exist: {input_path}", 400)

        document_service = DocumentService()
        result = document_service.extract_patterns_from_documents(input_path, files)

        if result["success"]:
            return success_response(
                {
                    "message": "Pattern extraction completed successfully",
                    "patterns_saved": result["patterns_saved"],
                    "headings_analyzed": result["headings_analyzed"],
                    "files_processed": result["files_processed"],
                    "analysis_metadata": result["analysis_metadata"],
                }
            )
        else:
            return error_response(
                "Pattern extraction failed",
                500,
                {"errors": result.get("errors", []), "details": result},
            )

    except Exception as e:
        current_app.logger.error(f"Error extracting patterns: {str(e)}")
        return error_response(f"Error extracting patterns: {str(e)}", 500)
