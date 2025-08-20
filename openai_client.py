import os
import logging
from typing import Dict, Any, Optional
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIClient:
    """OpenAI API client with support for both standard and Azure endpoints"""
    
    def __init__(self, api_config: Optional[Dict[str, Any]] = None):
        """Initialize OpenAI client based on configuration"""
        self.api_config = api_config or self._get_default_config()
        self.client = None
        self.model = None
        
        # Initialize the appropriate client
        if self.api_config.get('use_azure', False):
            self._initialize_azure_client()
        else:
            self._initialize_standard_client()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default OpenAI configuration"""
        return {
            'use_azure': False,
            'api_key': os.getenv('OPENAI_API_KEY'),
            'model': 'gpt-4.1-2025-04-14'
        }
    
    def _initialize_azure_client(self):
        """Initialize Azure OpenAI client"""
        try:
            api_key = self.api_config.get('api_key') or os.getenv('AZURE_API_KEY')
            if not api_key:
                raise ValueError("AZURE_API_KEY not found in configuration or environment variables")
            
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=self.api_config.get('api_version', 'gpt-4.1'),
                azure_endpoint=self.api_config.get('azure_endpoint')
            )
            self.model = self.api_config.get('model', 'gpt-4.1')
            logger.info("Initialized Azure OpenAI client")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
    
    def _initialize_standard_client(self):
        """Initialize standard OpenAI client"""
        try:
            api_key = self.api_config.get('api_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in configuration or environment variables")
            
            self.client = OpenAI(api_key=api_key)
            self.model = self.api_config.get('model', 'gpt-4.1-2025-04-14')
            logger.info("Initialized standard OpenAI client")
            
        except Exception as e:
            logger.error(f"Failed to initialize standard OpenAI client: {e}")
            raise
    
    def generate_pydantic_models(self, field_config: Dict[str, Any]) -> str:
        """Generate Pydantic models using OpenAI"""
        use_case = field_config.get('use_case', 'Document Analysis')
        description = field_config.get('description', 'Extract structured information from documents')
        
        prompt = f"""
You are an expert Python developer specializing in Pydantic models. Based on the provided field configuration, generate complete Pydantic models with proper imports, enums, and validation.

USE CASE CONTEXT: {use_case}
DESCRIPTION CONTEXT: {description}

Use the above context to understand the domain and purpose when generating models.

Field Configuration:
{self._format_config(field_config)}

Requirements:
1. Generate all necessary imports at the top
2. Create enum classes ONLY for fields with enum_values (skip if enum_values is null)
3. Create the main Pydantic model class
4. Use proper type hints (str, int, float, bool, List[enum], etc.)
5. Use field descriptions exactly as provided - do not modify or optimize them
6. Add field validators for list fields where needed
7. Handle both single enums and list[enum] types properly
8. Use proper enum inheritance (str, Enum)
9. Consider the use case context when structuring the models

IMPORTANT: Use field descriptions exactly as provided in the configuration. Do not modify, shorten, or optimize them.

Generate ONLY the Python code, no explanations or markdown formatting.

Example structure:
```python
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import List, Optional

class Domain(str, Enum):
    HEALTHCARE = "Healthcare & wellbeing"
    AUTOMOTIVE = "Automotive"
    CONSTRUCTION = "Construction"
    MANUFACTURING = "Manufacturing"
    FINANCE = "Finance"

class AiField(str, Enum):
    GENERATIVE_AI = "Generative AI"
    MACHINE_LEARNING = "Machine learning"
    COMPUTER_VISION = "Computer vision & image processing"

class MainModel(BaseModel):
    company_name: str = Field(..., description="The name of the company")
    domain: Domain = Field(..., description="The primary industry domain")
    ai_field: AiField = Field(..., description="The primary AI field")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
```

CRITICAL ENUM NAMING RULES:
1. Use clear, readable enum member names (HEALTHCARE, not HEALTHCAREW)
2. Keep enum names concise but descriptive
3. Use UPPER_CASE with underscores for multi-word concepts
4. Map readable names to full descriptive values
5. Never truncate or abbreviate enum member names randomly

Make sure all enum values are properly formatted and all field types are correct based on the configuration.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert Python developer. Generate clean, well-structured Pydantic models based on the provided specifications."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                #max_tokens=4000,
                max_completion_tokens = 4000,
                #temperature=0.0
            )
            
            logger.info("Successfully generated Pydantic models using OpenAI")
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI model generation failed: {e}")
            raise
    
    def _format_config(self, field_config: Dict[str, Any]) -> str:
        """Format field configuration for the prompt"""
        import json
        return json.dumps(field_config, indent=2)
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get information about the OpenAI client configuration"""
        return {
            "provider": "OpenAI",
            "model": self.model,
            "use_azure": self.api_config.get('use_azure', False),
            "endpoint_type": "Azure" if self.api_config.get('use_azure', False) else "Standard"
        }