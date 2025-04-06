import os
import tenacity
from typing import Optional, Dict, Any, List, Union
from anthropic import Anthropic

DEFAULT_MODEL = "claude-3-5-sonnet-20240620"
DEFAULT_MAX_TOKENS = 1600
DEFAULT_TEMPERATURE = 0

class AnthropicClient:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        """
        Initializes the AnthropicClient with the provided API key and model.
        If the API key is not provided, it attempts to read it from the environment variable ANTHROPIC_API_KEY.

        Args:
            api_key (Optional[str]): The API key for the Anthropic service.
            model (str): The model to use for the Anthropic service.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in the ANTHROPIC_API_KEY environment variable.")
        
        self.model = model
        self.client = Anthropic(api_key=self.api_key)

    def _create_message_payload(self, 
                              content: str,
                              max_tokens: int = DEFAULT_MAX_TOKENS,
                              temperature: float = DEFAULT_TEMPERATURE) -> Dict[str, Any]:
        """
        Creates the payload for the message to be sent to the Anthropic service.

        Args:
            content (str): The text content of the message.
            max_tokens (int): The maximum number of tokens for the response.
            temperature (float): The temperature setting for the response.

        Returns:
            Dict[str, Any]: The payload for the message.
        """
        message_content = []

        # Add text content
        message_content.append({
            "type": "text",
            "text": content
        })

        return {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": message_content
                }
            ]
        }

    @tenacity.retry(wait=tenacity.wait_random_exponential(multiplier=0.3, exp_base=3, max=90),
                    stop=tenacity.stop_after_attempt(6),
                    reraise=True)
    def send_message(self, 
                    content: str, 
                    max_tokens: int = DEFAULT_MAX_TOKENS, 
                    temperature: float = DEFAULT_TEMPERATURE,
                    json_eval: bool = False) -> Dict[str, Any]:
        """
        Sends a message to the Anthropic service and returns the response.

        Args:
            content (str): The text content of the message.
            max_tokens (int): The maximum number of tokens for the response.
            temperature (float): The temperature setting for the response.
            json_eval (bool): Whether to evaluate the result as JSON.

        Returns:
            Dict[str, Any]: A dictionary containing the status and the result of the response.
        """
        payload = self._create_message_payload(content, max_tokens, temperature)
        try:
            message = self.client.messages.create(**payload)
            result_text = message.content[0].text

            if json_eval:
                result = self._safe_eval(result_text)
            else:
                result = result_text
            
            return result
        except Exception as e:
            return {'status': False, 'error': str(e)}

    @staticmethod
    def _safe_eval(expression: str) -> Any:
        """
        Safely evaluates a string expression and returns the result.
        Uses ast.literal_eval for safe evaluation.

        Args:
            expression (str): The expression to evaluate.

        Returns:
            Any: The evaluated result.
        """
        import ast
        try:
            return ast.literal_eval(expression)
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {expression}. Error: {e}")