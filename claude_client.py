import os
import json
from typing import Dict, Any, List
import anthropic
from dotenv import load_dotenv

load_dotenv()

class ClaudeClient:
    def __init__(self):
        self.api_key = os.getenv('CLAUDE_API_KEY')
        if not self.api_key:
            raise ValueError("CLAUDE_API_KEY not found in environment variables")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-20250514"
        self.temperature = 0.0  # Zero temperature for consistent output
    
    def generate_pydantic_models(self, field_config: Dict[str, Any]) -> str:
        """Generate Pydantic models based on field configuration"""
        
        use_case = field_config.get('use_case', 'Document Analysis')
        description = field_config.get('description', 'Extract structured information from documents')
        
        prompt = f"""
You are an expert Python developer specializing in Pydantic models. Based on the provided field configuration, generate complete Pydantic models with proper imports, enums, and validation.

USE CASE CONTEXT: {use_case}
DESCRIPTION CONTEXT: {description}

Use the above context to understand the domain and purpose when generating models.

Field Configuration:
{json.dumps(field_config, indent=2)}

CRITICAL FIELD NAMING REQUIREMENTS:
1. Convert all field names from the configuration to snake_case format
2. Replace spaces and hyphens with underscores, convert to lowercase
3. DO NOT use field aliases - the Python field names must match the JSON keys exactly
4. This ensures the extracted JSON can be validated without field name mismatches

Requirements:
1. Generate all necessary imports at the top
2. Create enum classes ONLY for fields with enum_values (skip if enum_values is null)
3. Create the main Pydantic model class with snake_case field names
4. Use proper type hints (str, int, float, bool, List[enum], etc.)
5. Use field descriptions exactly as provided - do not modify or optimize them
6. Add field validators for list fields where needed
7. Handle both single enums and list[enum] types properly
8. Use proper enum inheritance (str, Enum)
9. Consider the use case context when structuring the models
10. NEVER use aliases - use consistent snake_case field names throughout

IMPORTANT: 
- Use field descriptions exactly as provided in the configuration
- Generate ONLY clean Python code with no markdown formatting
- All field names must be snake_case to match JSON output

Example structure:
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

CRITICAL ENUM NAMING RULES:
1. Use clear, readable enum member names (HEALTHCARE, not HEALTHCAREW)
2. Keep enum names concise but descriptive
3. Use UPPER_CASE with underscores for multi-word concepts
4. Map readable names to full descriptive values
5. Never truncate or abbreviate enum member names randomly

Make sure all enum values are properly formatted and all field types are correct based on the configuration.
"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
