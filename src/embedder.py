import json
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai

from src.llm.api import init_genai_client

# Load environment variables
load_dotenv()


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load and parse a JSONL file.

    Args:
        file_path (str): Path to the JSONL file

    Returns:
        List[Dict[str, Any]]: List of parsed JSON objects

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON parsing fails
    """
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"JSONL file not found: {file_path}")


def extract_texts_from_jsonl(
    jsonl_data: List[Dict[str, Any]], content_field: str = "content"
) -> List[str]:
    """
    Extract text content from JSONL data.

    Args:
        jsonl_data (List[Dict[str, Any]]): Parsed JSONL data
        content_field (str): Field name containing the text content

    Returns:
        List[str]: List of text contents
    """
    texts = []
    for item in jsonl_data:
        if content_field in item and item[content_field]:
            texts.append(str(item[content_field]).strip())

    # If no texts were found using the requested field, try some common alternate fields
    if not texts:
        alternate_fields = ["question", "answer", "text", "body"]
        for field in alternate_fields:
            if field == content_field:
                continue
            alt_texts = []
            for item in jsonl_data:
                if field in item and item[field]:
                    alt_texts.append(str(item[field]).strip())
            if alt_texts:
                print(
                    f"No text found for field '{content_field}', falling back to '{field}'"
                )
                return alt_texts

    return texts


def create_embeddings(
    client: genai.Client,
    texts: List[str],
    model: str = "gemini-embedding-001",
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: Optional[int] = 1536,
    title: Optional[str] = None,
) -> Any:
    """
    Create embeddings for given texts using Google GenAI.

    Args:
        client (genai.Client): Initialized GenAI client
        texts (List[str]): List of texts to embed
        model (str): Model name for embeddings. Defaults to "gemini-embedding-001"
        task_type (str): Task type for embeddings. Defaults to "RETRIEVAL_DOCUMENT"
        output_dimensionality (Optional[int]): Output dimension size. Defaults to 1536
        title (Optional[str]): Optional title for the embedding task

    Returns:
        Any: Embedding response from the API

    Raises:
        Exception: If embedding creation fails
    """
    config: Dict[str, Any] = {
        "task_type": task_type,
    }

    if output_dimensionality is not None:
        config["output_dimensionality"] = output_dimensionality

    if title is not None:
        config["title"] = title

    try:
        response = client.models.embed_content(
            model=model,
            contents=texts,  # type: ignore
            config=config,  # type: ignore
        )
        return response
    except Exception as e:
        raise Exception(f"Failed to create embeddings: {str(e)}")


def embed_single_text(
    client: genai.Client,
    text: str,
    model: str = "gemini-embedding-001",
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: Optional[int] = 1536,
    title: Optional[str] = None,
) -> Any:
    """
    Create embedding for a single text.

    Args:
        client (genai.Client): Initialized GenAI client
        text (str): Text to embed
        model (str): Model name for embeddings. Defaults to "gemini-embedding-001"
        task_type (str): Task type for embeddings. Defaults to "RETRIEVAL_DOCUMENT"
        output_dimensionality (Optional[int]): Output dimension size. Defaults to 1536
        title (Optional[str]): Optional title for the embedding task

    Returns:
        Any: Embedding response from the API
    """
    return create_embeddings(
        client=client,
        texts=[text],
        model=model,
        task_type=task_type,
        output_dimensionality=output_dimensionality,
        title=title,
    )


def get_embedding_vector(response: Any, index: int = 0) -> List[float]:
    """
    Extract embedding vector from API response.

    Args:
        response: API response containing embeddings
        index (int): Index of embedding to extract. Defaults to 0

    Returns:
        List[float]: Embedding vector

    Raises:
        IndexError: If index is out of range
        AttributeError: If response format is unexpected
    """
    try:
        return response.embeddings[index].values
    except (IndexError, AttributeError) as e:
        raise Exception(f"Failed to extract embedding vector: {str(e)}")


def embed_qa_file(
    qa_file_path: str,
    output_file_path: Optional[str] = None,
    model: str = "gemini-embedding-001",
    task_type: Optional[str] = None,
    output_dimensionality: Optional[int] = 1536,
) -> str:
    """
    Create embeddings for Q&A file and save to embedded format.

    Args:
        qa_file_path (str): Path to Q&A JSON file
        output_file_path (str): Output path for embedded Q&A file
        model (str): Model name for embeddings
        task_type (str): Task type for embeddings (defaults to config.EMBEDDING_TASK_TYPE)
        output_dimensionality (Optional[int]): Output dimension size

    Returns:
        str: Path to the output embedded file

    Raises:
        FileNotFoundError: If Q&A file doesn't exist
        Exception: If embedding process fails
    """
    if task_type is None:
        try:
            from .config import config

            task_type = config.EMBEDDING_TASK_TYPE
        except ImportError:
            task_type = "RETRIEVAL_DOCUMENT"  # fallback
    if not os.path.exists(qa_file_path):
        raise FileNotFoundError(f"Q&A file not found: {qa_file_path}")

    # Enforce JSONL-only Q&A input
    if not qa_file_path.lower().endswith(".jsonl"):
        raise ValueError(
            "embed_qa_file expects a .jsonl (newline-delimited JSON) Q&A file"
        )

    # Generate output file path if not provided - use .jsonl for embedded outputs
    if output_file_path is None:
        base_name = os.path.splitext(qa_file_path)[0]
        output_file_path = f"{base_name}_embedded.jsonl"

    try:
        # Load Q&A data from JSONL (newline-delimited JSON)
        qa_data: List[dict] = []
        with open(qa_file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    qa_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")

        print(f"📚 Loading {len(qa_data)} Q&A pairs from {qa_file_path}")

        # Initialize client
        client = init_genai_client()

        # Extract questions for embedding
        questions = []
        valid_qa_items = []

        for qa_item in qa_data:
            if "question" in qa_item and qa_item["question"]:
                questions.append(str(qa_item["question"]).strip())
                valid_qa_items.append(qa_item)

        if not questions:
            raise ValueError("No valid questions found in Q&A file")

        print(f"🔧 Creating embeddings for {len(questions)} questions...")

        # Create embeddings in batches
        batch_size = 10
        embedded_qa_data = []

        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i : i + batch_size]
            batch_qa_items = valid_qa_items[i : i + batch_size]

            print(
                f"   Processing batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}"
            )

            try:
                # Create embeddings for batch
                response = create_embeddings(
                    client=client,
                    texts=batch_questions,
                    model=model,
                    task_type=task_type,
                    output_dimensionality=output_dimensionality,
                    title=f"Q&A embeddings batch {i//batch_size + 1}",
                )

                # Process each item in batch
                for j, qa_item in enumerate(batch_qa_items):
                    try:
                        embedding = get_embedding_vector(response, j)

                        # Create embedded Q&A item
                        embedded_item = {
                            "question": qa_item["question"],
                            "answer": qa_item.get("answer", ""),
                            "embedding": embedding,
                            "embedding_dimensions": len(embedding),
                        }

                        # Add any additional fields
                        for key, value in qa_item.items():
                            if key not in ["question", "answer"]:
                                embedded_item[key] = value

                        embedded_qa_data.append(embedded_item)

                    except Exception as e:
                        print(f"⚠️  Error embedding question {i+j+1}: {e}")
                        # Add item without embedding
                        error_item = qa_item.copy()
                        error_item["embedding_error"] = str(e)
                        embedded_qa_data.append(error_item)

            except Exception as e:
                print(f"⚠️  Error processing batch {i//batch_size + 1}: {e}")
                # Add items without embeddings
                for qa_item in batch_qa_items:
                    error_item = qa_item.copy()
                    error_item["embedding_error"] = str(e)
                    embedded_qa_data.append(error_item)

        # Save embedded Q&A data as JSONL (newline-delimited) so it can be read incrementally
        print(f"💾 Saving embedded Q&A data to {output_file_path}")
        with open(output_file_path, "w", encoding="utf-8") as f:
            for item in embedded_qa_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Show statistics
        successful_embeddings = sum(
            1 for item in embedded_qa_data if "embedding" in item
        )
        failed_embeddings = len(embedded_qa_data) - successful_embeddings

        print(f"✅ Embedding complete!")
        print(f"   Successfully embedded: {successful_embeddings}")
        print(f"   Failed embeddings: {failed_embeddings}")
        print(f"   Output file: {output_file_path}")

        return output_file_path

    except Exception as e:
        raise Exception(f"Failed to embed Q&A file: {str(e)}")


def process_jsonl_embeddings(
    file_path: str,
    output_file: Optional[str] = None,
    batch_size: int = 10,
    content_field: str = "content",
) -> List[Dict[str, Any]]:
    """
    Process a JSONL file and create embeddings for all text content.

    Args:
        file_path (str): Path to the JSONL file
        output_file (Optional[str]): Path to save embeddings (optional)
        batch_size (int): Number of texts to process in each batch
        content_field (str): Field name containing the text content

    Returns:
        List[Dict[str, Any]]: List of items with embeddings added

    Raises:
        Exception: If processing fails
    """
    try:
        # Initialize client
        client = init_genai_client()

        # Load JSONL data
        print(f"Loading JSONL file: {file_path}")
        jsonl_data = load_jsonl_file(file_path)
        print(f"Loaded {len(jsonl_data)} items")

        # Extract texts
        texts = extract_texts_from_jsonl(jsonl_data, content_field)
        print(f"Extracted {len(texts)} text items")

        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_data = jsonl_data[i : i + batch_size]

            print(
                f"Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}"
            )

            # Create embeddings for batch
            response = create_embeddings(
                client=client,
                texts=batch_texts,
                title=f"Document embeddings batch {i // batch_size + 1}",
            )

            # Add embeddings to original data
            for j, item in enumerate(batch_data):
                try:
                    embedding = get_embedding_vector(response, j)
                    item_with_embedding = item.copy()
                    item_with_embedding["embedding"] = embedding
                    item_with_embedding["embedding_dimensions"] = len(embedding)
                    results.append(item_with_embedding)
                except Exception as e:
                    print(f"Error processing item {i + j}: {e}")
                    # Add item without embedding
                    item_with_embedding = item.copy()
                    item_with_embedding["embedding"] = None
                    item_with_embedding["embedding_error"] = str(e)
                    results.append(item_with_embedding)

        # Save results if output file is specified
        if output_file:
            print(f"Saving results to: {output_file}")
            with open(output_file, "w", encoding="utf-8") as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return results

    except Exception as e:
        raise Exception(f"Failed to process JSONL embeddings: {str(e)}")


def main():
    """
    Demo function showing how to use the embedding functions with JSONL files.
    """
    try:
        # Import configuration: must be run as a package so relative imports work.
        # Run this script with: python -m src.embedder
        try:
            from .config import config, paths
        except Exception as e:
            raise RuntimeError(
                "embedder must be executed as a package. Start it with: python -m src.embedder"
            ) from e

        # Example 1: Process the provided JSONL file
        jsonl_file_path = str(
            paths.PROCESSED_DATA_DIR
            / r"D:\Reecotech\TEST\data\qa_samples\Bo_cau_hoi_BAS.jsonl"
        )
        output_file = str(
            paths.EMBEDDINGS_DIR
            / r"D:\Reecotech\TEST\data\qa_samples\Bo_cau_hoi_BAS_embeddings.jsonl"
        )

        if os.path.exists(jsonl_file_path):
            print("Processing JSONL file with embeddings...")
            output_path = embed_qa_file(
                qa_file_path=jsonl_file_path, output_file_path=output_file
            )

            # Load the embedded file to show results
            results = load_jsonl_file(output_path)
            print(f"\nProcessed {len(results)} items")

            # Show sample results
            if results:
                print("\nSample result:")
                sample = results[0]
                print(f"Content: {sample.get('content', '')[:100]}...")
                if "embedding" in sample and sample["embedding"]:
                    print(
                        f"Embedding dimensions: {sample.get('embedding_dimensions', 'N/A')}"
                    )
                    print(f"First 5 embedding values: {sample['embedding'][:5]}")
                else:
                    print(f"Embedding error: {sample.get('embedding_error', 'N/A')}")
        else:
            print(f"JSONL file not found: {jsonl_file_path}")

            # Example 2: Create embeddings for sample texts
            print("\nRunning sample embedding demo...")
            client = init_genai_client()

            sample_texts = [
                "Thiết bị lấy mẫu nước tích hợp IWS III 2.5 l",
                "Lấy mẫu nước tích hợp theo độ sâu hoặc thời gian",
                "Nguồn cung cấp điện Lithium sắt phosphate",
            ]

            response = create_embeddings(
                client=client,
                texts=sample_texts,
                title="Vietnamese Water Sampler",
                model=config.EMBEDDING_MODEL,
                task_type=config.EMBEDDING_TASK_TYPE,
                output_dimensionality=config.EMBEDDING_DIMENSIONS,
            )

            print("Embedding response:")
            for i, text in enumerate(sample_texts):
                try:
                    vector = get_embedding_vector(response, i)
                    print(f"\nText {i+1}: {text}")
                    print(f"Embedding dimensions: {len(vector)}")
                    print(f"First 5 values: {vector[:5]}")
                except Exception as e:
                    print(f"Error extracting embedding {i}: {e}")

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
