import logging
import sys
from typing import Any, Dict, List, Optional
from functools import wraps
from logging.handlers import RotatingFileHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain_core.messages import BaseMessage
from logging import Handler


# Create logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear any existing handlers
logger.handlers = []

# Add file handler
file_handler = RotatingFileHandler(
    'app.log',  
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
    async def wrapper(*args, **kwargs):
        query = args[0] if args else kwargs.get('query', 'No query provided')
        logger.info(f"RAG Query - Input: {query}")
        result = await func(*args, **kwargs)  # Await the coroutine
        logger.info(f"RAG Query - Found {len(result)} results")
        return result
    return wrapper



class NiceGuiLogElementCallbackHandler(BaseCallbackHandler):
    """Callback Handler that writes to a log element in gui"""

    def __init__(self, log_element = None) -> None:
        """Initialize callback handler."""
        self.log = log_element
        self._token_buffer = ""

    def set_log_element(self, log_element):
        """Set or update the log element after initialization"""
        self.log_element = log_element

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when LLM starts."""
        if self.log_element:
            self.log_element.push("\n> Starting chat model call...")

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any) -> None:
        """Run when Chat Model starts."""
        if self.log_element:
            self.log_element.push("\n> Starting chat model call...")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        # Accumulate tokens in the buffer
        self._token_buffer += token

        # Check if we have a complete sentence or meaningful chunk
        if any(p in self._token_buffer for p in ['.', '!', '?', '\n']) or len(self._token_buffer) > 50:
            if self.log_element:
                self.log_element.push(self._token_buffer)
            self._token_buffer = ""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends."""
        # Flush any remaining tokens
        if self._token_buffer:
            self.log.push(self._token_buffer)
            self._token_buffer = ""
        if self.log_element:
            self.log_element.push("\n> LLM call finished.")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Run when LLM errors."""
        if self.log_element:
            self.log_element.push(f"\n> LLM error: {str(error)}")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain starts."""
        if self.log_element:
            self.log_element.push(f"\n\n> Entering new chain...")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends."""
        if self.log_element:
            self.log_element.push("\n> Finished chain.")

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Run when chain errors."""
        if self.log_element:
            self.log_element.push(f"\n> Chain error: {str(error)}")

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""
        if self.log_element:
            self.log_element.push(text)

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        if self.log_element:
            self.log_element.push(action.log)

    def on_tool_end(self,
                    output: str,
                    observation_prefix: Optional[str] = None,
                    llm_prefix: Optional[str] = None,
                    **kwargs: Any,
                    ) -> None:
        """If not the final action, print out observation."""
        if observation_prefix is not None:
            self.log_element.push(f'\n{observation_prefix}')
        self.log_element.push(output)
        if llm_prefix is not None:
            self.log_element.push(f'\n{llm_prefix}')

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        if self.log_element:
            self.log_element.push(finish.log)

class NiceGuiLogHandler(Handler):
    """Custom logging handler that writes to a NiceGUI log element"""
    def __init__(self, log_element=None):
        super().__init__()
        self.log_element = log_element
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        if self.log_element:
            try:
                msg = self.format(record)
                self.log_element.push(f"{msg}\n")
            except Exception:
                self.handleError(record)

    def set_log_element(self, log_element):
        """Set or update the log element after initialization"""
        self.log_element = log_element

# Create the handler with no initial log element
nicegui_handler = NiceGuiLogHandler()

# Add it to the logger
logger.addHandler(nicegui_handler)

