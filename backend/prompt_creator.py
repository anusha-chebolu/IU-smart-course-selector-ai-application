import os
from typing import Optional

class PromptGenerator:
    def __init__(self, base_prompt_file: str = "base_prompt.txt", examples_file: str = "few_shots.txt"):
        self.base_prompt_file = base_prompt_file
        self.examples_file = examples_file
        self._base_prompt = None
        self._examples = None
    
    def _read_file(self, file_path: str) -> str:
        """Reads the content of a file and returns it as a string."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} was not found.")
        except IOError:
            raise IOError(f"An error occurred while reading the file {file_path}.")
    
    @property
    def base_prompt(self) -> str:
        """Lazily loads and returns the base prompt from the file."""
        if self._base_prompt is None:
            self._base_prompt = self._read_file(self.base_prompt_file)
        return self._base_prompt
    
    @property
    def examples(self) -> str:
        """Lazily loads and returns the examples from the file."""
        if self._examples is None:
            self._examples = self._read_file(self.examples_file)
        return self._examples
    
    def generate_full_prompt(self, prompt_input: str) -> str:
        """
        Generates the full prompt by combining the base prompt, examples,
        extra instructions, and the provided prompt input.
        
        Args:
            prompt_input (str): The input to be appended to the prompt.
            extra_instructions (Optional[str]): Additional instructions to be included in the prompt.
        
        Returns:
            str: The complete prompt ready for use.
        """
        final_prompt = f"{self.base_prompt}\n\n"
        
        final_prompt += f"{self.examples}\n {prompt_input}\n\n"
        
        return final_prompt