import json
from typing import List, Optional, Dict
import requests

class VectorizerClient:
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    def get_embeddings(self, texts: Optional[List[str]] = None) -> Dict:
        if not texts:
            raise ValueError("At least one text string must be provided for embedding")
        
        data = {}
        
        if texts:
            data["texts"] = texts

        try:
            response = requests.post(
                self.endpoint_url,
                data=json.dumps(data),
                headers={"Content-Type": "application/json"}
            )

            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error making request to embedding api endpoint: {str(e)}")