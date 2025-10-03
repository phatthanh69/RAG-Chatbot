import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


def init_genai_client() -> genai.Client:
    """
    Initialize and return a Google GenAI client for embeddings.

    Returns:
        genai.Client: Configured GenAI client

    Raises:
        ValueError: If required environment variables are not set
    """
    use_vertexai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
    api_key = os.getenv("GOOGLE_API_KEY")

    if use_vertexai:

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION")

        if not project:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable is required for Vertex AI"
            )
        if not location:
            raise ValueError(
                "GOOGLE_CLOUD_LOCATION environment variable is required for Vertex AI"
            )

        return genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )
    else:

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for Google AI API"
            )

        return genai.Client(api_key=api_key)
