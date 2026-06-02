"""
Response helper functions for consistent API responses
"""

from datetime import datetime, timedelta, timezone

from flask import jsonify

# Timezone configuration for Vietnam (UTC+7)
VIETNAM_TIMEZONE = timezone(timedelta(hours=7))


def get_vietnam_time():
    """
    Get current time in Vietnam timezone (UTC+7)

    Returns:
        datetime: Current time in Vietnam timezone
    """
    return datetime.now(VIETNAM_TIMEZONE)


def success_response(data=None, status_code=200):
    """
    Create a standardized success response

    Args:
        data: Response data
        status_code: HTTP status code

    Returns:
        JSON response with success format
    """
    response = {
        "success": True,
        "timestamp": get_vietnam_time().isoformat(),
        "status_code": status_code,
    }

    if data is not None:
        if isinstance(data, dict):
            response.update(data)
        else:
            response["data"] = data

    return jsonify(response), status_code


def error_response(message, status_code=400):
    """
    Create a standardized error response

    Args:
        message: Error message or dict with error details
        status_code: HTTP status code

    Returns:
        JSON response with error format
    """
    error_data = {
        "success": False,
        "timestamp": get_vietnam_time().isoformat(),
        "status_code": status_code,
    }

    if isinstance(message, dict):
        error_data.update(message)
    else:
        error_data["message"] = message
        error_data["error"] = str(message)

    return jsonify(error_data), status_code


def paginated_response(data, page, per_page, total_count, status_code=200):
    """
    Create a paginated response

    Args:
        data: List of items for current page
        page: Current page number
        per_page: Items per page
        total_count: Total number of items
        status_code: HTTP status code

    Returns:
        JSON response with pagination info
    """
    response = {
        "success": True,
        "timestamp": get_vietnam_time().isoformat(),
        "status_code": status_code,
        "data": data,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_count": total_count,
            "total_pages": (total_count + per_page - 1) // per_page,
        },
    }

    return jsonify(response), status_code
