import logging
from functools import wraps
import sys
from logging.handlers import RotatingFileHandler

# Create logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear any existing handlers
logger.handlers = []

# Add file handler
file_handler = RotatingFileHandler(
    'app.log',  # This will create/append to app.log in your project directory
    maxBytes=1024 * 1024,  # 1MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Add stdout handler
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stdout_handler)

# Ensure handlers propagate
logger.propagate = False

def log_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Completed {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def log_rag_query(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        query = args[0] if args else kwargs.get('query', 'No query provided')
        logger.info(f"RAG Query - Input: {query}")
        result = func(*args, **kwargs)
        logger.info(f"RAG Query - Found {len(result)} results")
        return result
    return wrapper
