"""Markdown document service integrating the Markdown extractor with the RAG chatbot."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.markdown_extractor import process_markdown_file

# Vietnam timezone
VIETNAM_TIMEZONE = timezone(timedelta(hours=7))


def get_vietnam_now():
    """Get current time in Vietnam timezone"""
    return datetime.now(VIETNAM_TIMEZONE)


class MarkdownDocumentService:
    """Service class for processing and storing Markdown documents in the database."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process_and_store_markdown(
        self, file_path: str, clean_pdf_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a Markdown file and store its content in the database similar to PDF handling.

        Args:
            file_path: Path to the Markdown file.
            clean_pdf_name: Sanitized file name used when storing metadata.

        Returns:
            Dict describing the processing result.
        """
        try:
            file_path_obj = Path(file_path)

            if not file_path_obj.exists():
                return {"error": f"File does not exist: {file_path_obj}"}

            if clean_pdf_name is None:
                clean_pdf_name = file_path_obj.stem

            self.logger.info(f"Starting Markdown processing: {file_path_obj}")

            # Extract and chunk Markdown
            result = process_markdown_file(file_path_obj)

            if "error" in result:
                return result

            # Convert chunks into a database-compatible format
            chunks_for_db = []

            for i, chunk in enumerate(result["chunks"]):
                chunk_data = {
                    "chunk_index": i,
                    "content": chunk["content"],
                    "page_number": None,  # Markdown has no page information
                    "block_index": chunk.get("line_number", 0),
                    "bbox": None,  # Markdown has no bounding boxes
                    "font_info": None,  # Markdown has no font info
                    "heading_id": chunk.get("heading_id"),
                    "heading_title": chunk.get("heading_title"),
                    "heading_parent_id": chunk.get("heading_parent_id"),
                    "heading_level": chunk.get("heading_level"),
                    "is_heading": chunk.get("is_heading", False),
                    "clean_pdf_name": clean_pdf_name,  # Keep this for compatibility
                    "block_type": chunk.get("block_type", "text"),
                }
                chunks_for_db.append(chunk_data)

            self.logger.info(f"Created {len(chunks_for_db)} Markdown chunks")

            return {
                "success": True,
                "file_name": file_path_obj.name,
                "file_path": str(file_path_obj),
                "chunks": chunks_for_db,
                "total_chunks": len(chunks_for_db),
                "metadata": {
                    "source_type": "markdown",
                    "total_blocks": result["total_blocks"],
                    "processing_time": get_vietnam_now().isoformat(),
                },
            }

        except Exception as e:
            self.logger.error(f"Failed while processing Markdown file: {e}")
            return {"error": str(e)}

    def batch_process_markdown_directory(
        self, directory_path: str, file_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Batch-process every Markdown file inside a directory.

        Args:
            directory_path: Path to the directory.
            file_patterns: List of glob patterns to match (defaults to ["*.md", "*.markdown"]).

        Returns:
            Dict describing the batch processing results.
        """
        if file_patterns is None:
            file_patterns = ["*.md", "*.markdown"]

        directory = Path(directory_path)

        if not directory.exists() or not directory.is_dir():
            return {"error": f"Directory does not exist: {directory_path}"}

        # Discover every Markdown file
        md_files = []
        for pattern in file_patterns:
            md_files.extend(list(directory.glob(pattern)))

        if not md_files:
            return {"error": f"No Markdown files found in: {directory_path}"}

        results = []
        errors = []

        self.logger.info(f"Starting batch processing for {len(md_files)} Markdown files")

        for md_file in md_files:
            try:
                result = self.process_and_store_markdown(md_file)

                if "error" in result:
                    errors.append(f"{md_file.name}: {result['error']}")
                else:
                    results.append(
                        {
                            "file_name": md_file.name,
                            "chunks_count": result["total_chunks"],
                            "status": "success",
                        }
                    )

            except Exception as e:
                error_msg = f"{md_file.name}: {str(e)}"
                errors.append(error_msg)
                self.logger.error(f"Failed to process {md_file}: {e}")

        return {
            "total_files": len(md_files),
            "processed_files": len(results),
            "failed_files": len(errors),
            "results": results,
            "errors": errors,
        }

    def convert_markdown_to_jsonl(
        self,
        md_file_path: str,
        output_path: Optional[str] = None,
        include_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert a Markdown file to the JSONL format used by the pipeline.

        Args:
            md_file_path: Path to the Markdown file.
            output_path: Destination JSONL path.
            include_embeddings: Whether embeddings should be included alongside the content.

        Returns:
            Dict describing the conversion result.
        """
        try:
            # Process the Markdown file
            result = self.process_and_store_markdown(md_file_path)

            if "error" in result:
                return result

            # Prepare the output path
            md_path = Path(md_file_path)
            if output_path is None:
                output_path_obj = Path("data/processed") / f"{md_path.stem}_processed.jsonl"
            else:
                output_path_obj = Path(output_path)

            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Convert to JSONL format
            jsonl_entries = []

            for chunk in result["chunks"]:
                entry = {
                    "content": chunk["content"],
                    "meta": {
                        "source": "markdown",
                        "file_name": result["file_name"],
                        "chunk_index": chunk["chunk_index"],
                        "line_number": chunk.get("block_index", 0),
                        "heading_id": chunk.get("heading_id"),
                        "heading_title": chunk.get("heading_title"),
                        "heading_level": chunk.get("heading_level"),
                        "is_heading": chunk.get("is_heading", False),
                        "block_type": chunk.get("block_type", "text"),
                        "clean_pdf_name": chunk.get("clean_pdf_name"),  # Compatibility
                    },
                }

                if include_embeddings:
                    # TODO: Integrate with the embedding service
                    entry["embedding"] = None  # Placeholder

                jsonl_entries.append(entry)

            # Persist the JSONL file
            with open(output_path_obj, "w", encoding="utf-8") as f:
                for entry in jsonl_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            self.logger.info(f"Saved {len(jsonl_entries)} entries to: {output_path_obj}")

            return {
                "success": True,
                "output_file": str(output_path_obj),
                "total_entries": len(jsonl_entries),
                "file_size": output_path_obj.stat().st_size,
            }

        except Exception as e:
            self.logger.error(f"Failed to convert Markdown to JSONL: {e}")
            return {"error": str(e)}


def create_markdown_processing_script():
    """Create a CLI script that processes Markdown files."""

    script_content = '''#!/usr/bin/env python3
"""
CLI script to process Markdown files for the RAG Chatbot
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from markdown_document_service import MarkdownDocumentService


def main():
    parser = argparse.ArgumentParser(description="Process Markdown files for the RAG Chatbot")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Process single file
    process_parser = subparsers.add_parser('process', help='Process a single Markdown file')
    process_parser.add_argument('file_path', help='Path to the Markdown file')
    process_parser.add_argument('--output', '-o', help='Output directory')
    
    # Process directory
    batch_parser = subparsers.add_parser('batch', help='Process every Markdown file in a directory')
    batch_parser.add_argument('directory', help='Directory containing Markdown files')
    batch_parser.add_argument('--patterns', nargs='+', default=['*.md', '*.markdown'], help='List of file patterns to include')
    
    # Convert to JSONL
    convert_parser = subparsers.add_parser('convert', help='Convert a Markdown file to JSONL')
    convert_parser.add_argument('file_path', help='Path to the Markdown file')
    convert_parser.add_argument('--output', '-o', help='Output JSONL file path')
    convert_parser.add_argument('--embeddings', action='store_true', help='Generate embeddings for each chunk')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    service = MarkdownDocumentService()
    
    if args.command == 'process':
        print(f"Processing file: {args.file_path}")
        result = service.process_and_store_markdown(args.file_path)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return 1
        
        print(f"Success! Created {result['total_chunks']} chunks")
        
    elif args.command == 'batch':
        print(f"Processing directory: {args.directory}")
        result = service.batch_process_markdown_directory(args.directory, args.patterns)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return 1
        
        print(f"Completed: {result['processed_files']}/{result['total_files']} files successfully")
        
        if result['errors']:
            print("Errors:")
            for error in result['errors']:
                print(f"  - {error}")
    
    elif args.command == 'convert':
        print(f"Converting file: {args.file_path}")
        result = service.convert_markdown_to_jsonl(
            args.file_path, 
            args.output, 
            args.embeddings
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return 1
        
        print(f"Success! Created file: {result['output_file']}")
        print(f"Entries: {result['total_entries']}, Size: {result['file_size']} bytes")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

    script_path = Path("process_markdown.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    # Make executable on Unix systems
    try:
        import stat

        script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
    except Exception:
        pass

    return script_path


if __name__ == "__main__":
    # Demo usage
    service = MarkdownDocumentService()

    print("=== Markdown Document Service Demo ===")
    print("Service initialized successfully!")
    print()

    # Create the CLI script
    script_path = create_markdown_processing_script()
    print(f"Created CLI script: {script_path}")
    print()

    print("How to use:")
    print(f"  python {script_path} process <markdown_file>")
    print(f"  python {script_path} batch <directory>")
    print(f"  python {script_path} convert <markdown_file> --output <output.jsonl>")
