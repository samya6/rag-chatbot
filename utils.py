import hashlib


def get_file_hash(file_path: str) -> str:
    """
    Generate a unique hash for a file based on its content.
    Used for caching processed PDFs.

    Args:
        file_path (str): Path to the file

    Returns:
        str: MD5 hash string
    """

    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()