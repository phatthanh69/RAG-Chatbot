#!/usr/bin/env python3
"""
CLI Tool for Enhanced Markdown Document Service
Command-line interface to process Markdown files with full pipeline
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_markdown_service import EnhancedMarkdownDocumentService


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("markdown_processor.log"),
        ],
    )


def process_single_file(args) -> Dict[str, Any]:
    """Process a single Markdown file"""
    service = EnhancedMarkdownDocumentService()

    print(f"🚀 Processing file: {args.input}")
    print(f"📁 Output directory: {args.output}")
    print(f"📦 Max chunk size: {args.chunk_size}")
    print(f"🔮 Create embeddings: {args.embeddings}")
    print(f"💾 Store in database: {args.database}")
    print("-" * 60)

    result = service.process_markdown_file(
        file_path=args.input,
        output_dir=args.output,
        max_chunk_size=args.chunk_size,
        create_embeddings=args.embeddings,
        store_in_database=args.database,
    )

    return result


def process_batch(args) -> Dict[str, Any]:
    """Process batch of Markdown files"""
    service = EnhancedMarkdownDocumentService()

    print(f"🚀 Batch processing directory: {args.input}")
    print(f"📁 Output directory: {args.output}")
    print(f"🔍 File pattern: {args.pattern}")
    print(f"📦 Max chunk size: {args.chunk_size}")
    print(f"🔮 Create embeddings: {args.embeddings}")
    print(f"💾 Store in database: {args.database}")
    print("-" * 60)

    result = service.batch_process_markdown_directory(
        input_dir=args.input,
        output_dir=args.output,
        max_chunk_size=args.chunk_size,
        create_embeddings=args.embeddings,
        store_in_database=args.database,
        file_pattern=args.pattern,
    )

    return result


def convert_to_jsonl(args) -> Dict[str, Any]:
    """Convert Markdown to JSONL format"""
    service = EnhancedMarkdownDocumentService()

    print(f"🔄 Converting to JSONL: {args.input}")
    print(f"📄 Output file: {args.output}")
    print(f"📦 Max chunk size: {args.chunk_size}")
    print("-" * 60)

    result = service.convert_to_jsonl(
        file_path=args.input, output_file=args.output, max_chunk_size=args.chunk_size
    )

    return result


def print_result(result: Dict[str, Any], verbose: bool = False) -> None:
    """Print processing result"""
    print("\n" + "=" * 60)

    if "error" in result:
        print(f"❌ ERROR: {result['error']}")
        return

    # Single file result
    if "total_chunks" in result:
        print("✅ PROCESSING COMPLETED SUCCESSFULLY!")
        print(f"📄 File: {result.get('file_name', 'unknown')}")
        print(f"📊 Total blocks: {result.get('total_blocks', 0)}")
        print(f"📦 Total chunks: {result.get('total_chunks', 0)}")

        if result.get("embeddings_created"):
            print(f"🔮 Embeddings: {result.get('embeddings_count', 0)} created")

        if result.get("stored_in_database"):
            print(f"💾 Database: Document ID {result.get('document_id')}")

        if verbose:
            print(f"📁 Chunks file: {result.get('chunks_file')}")
            if result.get("embeddings_file"):
                print(f"📁 Embeddings file: {result.get('embeddings_file')}")

    # Batch result
    elif "processed_files" in result:
        print("✅ BATCH PROCESSING COMPLETED!")
        print(f"📊 Total files: {result.get('total_files', 0)}")
        print(f"✅ Processed successfully: {result.get('processed_files', 0)}")
        print(f"❌ Errors: {len(result.get('errors', []))}")

        if verbose and result.get("errors"):
            print("\n🚨 ERRORS:")
            for error in result["errors"]:
                print(
                    f"  - {error.get('file', 'unknown')}: {error.get('error', 'unknown error')}"
                )

    # JSONL conversion result
    elif "success" in result:
        if result["success"]:
            print("✅ JSONL CONVERSION COMPLETED!")
            print(f"📄 Input: {result.get('input_file')}")
            print(f"📄 Output: {result.get('output_file')}")
            print(f"📦 Total chunks: {result.get('total_chunks', 0)}")
            print(f"📏 File size: {result.get('file_size', 0):,} bytes")
        else:
            print(f"❌ CONVERSION FAILED: {result.get('error', 'unknown error')}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Enhanced Markdown Document Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file with full pipeline
  python markdown_cli.py process file.md

  # Process without embeddings
  python markdown_cli.py process file.md --no-embeddings

  # Process without database storage
  python markdown_cli.py process file.md --no-database

  # Batch process directory
  python markdown_cli.py batch /path/to/markdown/files

  # Convert to JSONL only
  python markdown_cli.py convert file.md output.jsonl

  # Custom chunk size
  python markdown_cli.py process file.md --chunk-size 1500
        """,
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process single file command
    process_parser = subparsers.add_parser(
        "process", help="Process a single Markdown file"
    )
    process_parser.add_argument("input", help="Path to Markdown file")
    process_parser.add_argument(
        "-o",
        "--output",
        default="data/processed",
        help="Output directory (default: data/processed)",
    )
    process_parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size (default: 1000)",
    )
    process_parser.add_argument(
        "--no-embeddings",
        action="store_false",
        dest="embeddings",
        help="Skip embedding creation",
    )
    process_parser.add_argument(
        "--no-database",
        action="store_false",
        dest="database",
        help="Skip database storage",
    )

    # Batch process command
    batch_parser = subparsers.add_parser(
        "batch", help="Batch process Markdown files in directory"
    )
    batch_parser.add_argument(
        "input", help="Path to directory containing Markdown files"
    )
    batch_parser.add_argument(
        "-o",
        "--output",
        default="data/processed",
        help="Output directory (default: data/processed)",
    )
    batch_parser.add_argument(
        "-p", "--pattern", default="*.md", help="File pattern to match (default: *.md)"
    )
    batch_parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size (default: 1000)",
    )
    batch_parser.add_argument(
        "--no-embeddings",
        action="store_false",
        dest="embeddings",
        help="Skip embedding creation",
    )
    batch_parser.add_argument(
        "--no-database",
        action="store_false",
        dest="database",
        help="Skip database storage",
    )

    # Convert to JSONL command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert Markdown to JSONL format"
    )
    convert_parser.add_argument("input", help="Path to Markdown file")
    convert_parser.add_argument("output", help="Output JSONL file path")
    convert_parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size (default: 1000)",
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    setup_logging(args.verbose)

    # Execute command
    try:
        if args.command == "process":
            if not Path(args.input).exists():
                print(f"❌ Input file not found: {args.input}")
                sys.exit(1)

            result = process_single_file(args)

        elif args.command == "batch":
            if not Path(args.input).exists():
                print(f"❌ Input directory not found: {args.input}")
                sys.exit(1)

            result = process_batch(args)

        elif args.command == "convert":
            if not Path(args.input).exists():
                print(f"❌ Input file not found: {args.input}")
                sys.exit(1)

            result = convert_to_jsonl(args)

        else:
            parser.print_help()
            sys.exit(1)

        # Print result
        print_result(result, args.verbose)

        # Set exit code based on success
        if "error" in result or (
            isinstance(result.get("success"), bool) and not result["success"]
        ):
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
