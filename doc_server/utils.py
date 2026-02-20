"""
Utility functions for doc-server.
"""


def _sanitize_input(input_str: str, max_length: int = 1000) -> str:
    """Sanitize user input to prevent injection attacks."""
    if not input_str or not input_str.strip():
        raise ValueError("Input cannot be empty")

    sanitized = input_str.replace("\x00", "")

    if len(sanitized) > max_length:
        raise ValueError(f"Input exceeds maximum length of {max_length} characters")

    suspicious_patterns = ["../", "..\\", "${", "`", "$(", "|"]
    for pattern in suspicious_patterns:
        if pattern in sanitized:
            raise ValueError(f"Input contains suspicious pattern: {pattern}")

    return sanitized
