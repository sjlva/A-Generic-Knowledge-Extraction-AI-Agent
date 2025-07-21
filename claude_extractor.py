import os
import json
import time
from typing import Dict, Any, Type, List
from claude_client import ClaudeClient
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaudeExtractor:
    """Claude AI powered data extraction"""
    
    def __init__(self, model_selection='claude-sonnet-4-20250514'):
        self.claude_client = ClaudeClient()
        self.model_selection = model_selection
        # Set the model on the claude client
        if hasattr(self.claude_client, 'model'):
            self.claude_client.model = model_selection
        
        self.max_tokens = 4000
        self.temperature = 0.0  # Zero temperature for maximum consistency
    
    def extract_data(self, document_content: str, extraction_prompt: str, 
                    model_class: Type, document_metadata: Dict[str, Any], 
                    additional_instructions: str = "") -> Dict[str, Any]:
        """Extract structured data from document using Claude AI"""
        
        # Parse additional instructions to extract context components
        extraction_purpose, document_type, custom_instructions = self._parse_additional_instructions(additional_instructions)
        
        # Prepare the extraction prompt with document content
        context_sections = []
        
        # Add extraction context prominently at the top
        if extraction_purpose.strip() or document_type.strip():
            context_sections.append("\n=== EXTRACTION CONTEXT ===\n\n")
            if extraction_purpose.strip():
                context_sections.append(f"The purpose of the extraction: {extraction_purpose.strip()}\n")
            if document_type.strip():
                context_sections.append(f"Document type: {document_type.strip()}.\n")
        
        # Add custom instructions if provided
        if custom_instructions.strip():
            context_sections.append(f"\nCUSTOM/ADDITIONAL EXTRACTION INSTRUCTIONS:\n{custom_instructions}\n")
        
        full_prompt = f"""

Extract the required information considering the context of EXTRACTION TASK, purpose of extraction, document type, and the EXTRACTION RULES. 
Return the extracted information as a valid JSON object that matches the specified schema.
Ensure all required fields are included and follow the exact field names and types specified.

{''.join(context_sections)}
{extraction_prompt}

IMPORTANT: Return ONLY the JSON object, no additional text, explanations, or markdown formatting.

================================================

Document to analyze:
Filename: {document_metadata.get('file_name', 'Unknown')}
Content:
{document_content}
"""
        
        try:
            logger.info(f"Extracting data from document: {document_metadata.get('file_name', 'Unknown')}")
            start_time = time.time()
            
            response = self.claude_client.client.messages.create(
                model=self.claude_client.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Claude extraction completed in {elapsed_time:.2f} seconds")
            
            # Parse the JSON response - Claude might wrap JSON in code blocks
            response_text = response.content[0].text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
            elif response_text.startswith('```'):
                response_text = response_text[3:]   # Remove ```
            
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove trailing ```
            
            response_text = response_text.strip()
            
            # Parse the JSON response
            extracted_json = json.loads(response_text)
            
            # Validate and create model instance
            try:
                model_instance = model_class(**extracted_json)
                logger.info(f"Successfully created model instance for {document_metadata.get('file_name')}")
                return model_instance.model_dump()
            
            except Exception as validation_error:
                logger.warning(f"Validation error for {document_metadata.get('file_name')}: {validation_error}")
                # Return raw JSON if validation fails
                return extracted_json
                
        except json.JSONDecodeError as json_error:
            logger.error(f"JSON parsing failed for document {document_metadata.get('file_name', 'Unknown')}")
            logger.error(f"Raw response: {response.content[0].text}")
            logger.error(f"Cleaned response: {response_text}")
            
            # Return fallback data structure with error info
            return self._create_fallback_data(model_class, document_metadata, f"JSON parsing failed: {str(json_error)}")
            
        except Exception as e:
            logger.error(f"Error extracting data from {document_metadata.get('file_name')}: {e}")
            # Return fallback data structure
            return self._create_fallback_data(model_class, document_metadata, str(e))
    
    def extract_batch(self, documents: List[Dict[str, Any]], extraction_prompt: str, 
                     model_class: Type, additional_instructions: str = "") -> List[Dict[str, Any]]:
        """Extract data from multiple documents using Claude"""
        
        results = []
        total_docs = len(documents)
        
        logger.info(f"Starting batch extraction for {total_docs} documents using Claude")
        
        for i, doc in enumerate(documents, 1):
            logger.info(f"Processing document {i}/{total_docs}: {doc.get('file_name')}")
            
            try:
                extracted_data = self.extract_data(
                    document_content=doc['text_content'],
                    extraction_prompt=extraction_prompt,
                    model_class=model_class,
                    document_metadata=doc,
                    additional_instructions=additional_instructions
                )
                
                # Add document metadata to results
                extracted_data['_document_metadata'] = {
                    'file_name': doc['file_name'],
                    'file_path': doc['file_path'],
                    'content_length': doc['content_length'],
                    'word_count': doc['word_count']
                }
                
                results.append(extracted_data)
                
            except Exception as e:
                logger.error(f"Failed to process document {doc.get('file_name')}: {e}")
                # Add fallback result
                fallback_data = self._create_fallback_data(model_class, doc, str(e))
                fallback_data['_document_metadata'] = {
                    'file_name': doc['file_name'],
                    'file_path': doc['file_path'],
                    'content_length': doc['content_length'],
                    'word_count': doc['word_count'],
                    'extraction_error': str(e)
                }
                results.append(fallback_data)
        
        logger.info(f"Claude batch extraction completed. Processed {len(results)} documents")
        return results
    
    def _create_fallback_data(self, model_class: Type, document_metadata: Dict[str, Any], error_message: str = None) -> Dict[str, Any]:
        """Create fallback data when extraction fails"""
        try:
            # Try to create a minimal instance with default values
            model_fields = model_class.model_fields
            fallback_data = {}
            
            for field_name, field_info in model_fields.items():
                if hasattr(field_info, 'default') and field_info.default is not None:
                    fallback_data[field_name] = field_info.default
                else:
                    # Provide basic fallback based on type
                    if 'str' in str(field_info.annotation):
                        fallback_data[field_name] = "n/a"
                    elif 'int' in str(field_info.annotation):
                        fallback_data[field_name] = 0
                    elif 'float' in str(field_info.annotation):
                        fallback_data[field_name] = 0.0
                    elif 'bool' in str(field_info.annotation):
                        fallback_data[field_name] = False
                    elif 'List' in str(field_info.annotation):
                        fallback_data[field_name] = []
                    else:
                        fallback_data[field_name] = "n/a"
            
            if error_message:
                fallback_data['error'] = error_message
                
            return fallback_data
            
        except Exception as e:
            logger.error(f"Error creating fallback data: {e}")
            return {
                "error": error_message or "Failed to extract data", 
                "file_name": document_metadata.get('file_name', 'Unknown'),
                "fallback_error": str(e)
            }
    
    def _parse_additional_instructions(self, additional_instructions: str) -> tuple[str, str, str]:
        """Parse additional instructions to extract purpose, document type, and custom instructions"""
        if not additional_instructions.strip():
            return "", "", ""
        
        # Split by double newlines to get separate instruction blocks
        instruction_blocks = additional_instructions.split('\n\n')
        
        extraction_purpose = ""
        document_type = ""
        custom_instructions = ""
        
        for block in instruction_blocks:
            # Check if this block contains the structured purpose/document type instruction
            if "The purpose of this extraction task is" in block and "Therefore, the document should be related to" in block:
                # Extract purpose and document type from the structured instruction
                import re
                purpose_match = re.search(r"The purpose of this extraction task is (.+?)\. Therefore, the document should be related to", block)
                doc_type_match = re.search(r"Therefore, the document should be related to (.+?)\. Do not attempt", block)
                
                if purpose_match:
                    extraction_purpose = purpose_match.group(1).strip()
                if doc_type_match:
                    document_type = doc_type_match.group(1).strip()
            else:
                # This is a custom instruction block
                if custom_instructions:
                    custom_instructions += "\n\n" + block.strip()
                else:
                    custom_instructions = block.strip()
        
        return extraction_purpose, document_type, custom_instructions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Claude model being used"""
        return {
            "model": self.model_selection,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "provider": "Claude"
        }