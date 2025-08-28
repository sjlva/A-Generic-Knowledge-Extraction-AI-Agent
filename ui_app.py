#!/usr/bin/env python3
"""
Unified Knowledge Extraction Agent UI
Combined configuration and extraction interface.
"""

import streamlit as st
import json
import os
import glob
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
import time
import logging

# Import our modules
from model_generator import ModelGenerator
from document_parser import DocumentParser
from openai_extractor import OpenAIExtractor
from claude_extractor import ClaudeExtractor

def get_api_config():
    """Get API configuration based on whether Azure endpoint is selected"""
    import os
    use_azure = st.session_state.get('use_azure', False)
    
    if use_azure:
        # Try secrets first, then fall back to environment variables
        try:
            azure_api_key = st.secrets["AZURE_API_KEY"]
            azure_open_ai_model_name = st.secrets["OPENAI_MODEL_NAME"]
            azure_endpoint = st.secrets["AZURE_ENDPOINT"]
            azure_api_version = st.secrets["AZURE_API_VERSION"]
        except:
            azure_api_key = os.getenv('AZURE_API_KEY')
            azure_open_ai_model_name = os.getenv("OPENAI_MODEL_NAME")
            azure_endpoint = os.getenv("AZURE_ENDPOINT")
            azure_api_version = os.getenv("AZURE_API_VERSION")
            
        return {
            'use_azure': True,
            'api_key': azure_api_key,
            'azure_endpoint': azure_endpoint,
            'api_version': azure_api_version,
            'model': azure_open_ai_model_name,  # Chat completion model
        }
    else:
        # Try secrets first, then fall back to environment variables
        try:
            openai_api_key = st.secrets["OPENAI_API_KEY"]
            open_ai_model_name = st.secrets["OPENAI_MODEL_NAME"]
        except:
            openai_api_key = os.getenv('OPENAI_API_KEY')
            openai_ai_model_name = os.getenv('OPENAI_MODEL_NAME')
            
        return {
            'use_azure': False,
            'api_key': openai_api_key,
            'model': open_ai_model_name,  # Chat completion model
        }

# Page configuration
st.set_page_config(
    page_title="Knowledge Extraction Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact, appealing design
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.3rem;
        color: #2e7d32;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2e7d32;
        padding-left: 1rem;
        font-weight: bold;
    }
    .field-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1f77b4 0%, #0d47a1 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .compact-input {
        margin-bottom: 0.5rem;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .results-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        border: 2px solid #28a745;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.15);
    }
    .results-header {
        font-size: 1.5rem;
        color: #28a745;
        font-weight: bold;
        margin-bottom: 1.5rem;
        text-align: center;
        border-bottom: 2px solid #28a745;
        padding-bottom: 0.5rem;
    }
    .result-summary {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .download-container {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #2196f3;
    }
    .certificate-green {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #155724;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 8px rgba(40, 167, 69, 0.2);
    }
    .certificate-red {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #721c24;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 8px rgba(220, 53, 69, 0.2);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Configuration"
    if 'fields' not in st.session_state:
        st.session_state.fields = []
    if 'use_case' not in st.session_state:
        st.session_state.use_case = ""
    if 'description' not in st.session_state:
        st.session_state.description = ""
    if 'main_model_name' not in st.session_state:
        st.session_state.main_model_name = ""
    if 'additional_instructions' not in st.session_state:
        st.session_state.additional_instructions = ""
    if 'extraction_purpose' not in st.session_state:
        st.session_state.extraction_purpose = ""
    if 'document_type' not in st.session_state:
        st.session_state.document_type = ""
    if 'custom_instructions' not in st.session_state:
        st.session_state.custom_instructions = ""
    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = []
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = None
    if 'rebuild_models' not in st.session_state:
        st.session_state.rebuild_models = False
    if 'model_generation_model' not in st.session_state:
        st.session_state.model_generation_model = 'claude-sonnet-4-20250514'
    if 'extraction_model' not in st.session_state:
        st.session_state.extraction_model = 'gpt-4.1-2025-04-14'
    if 'use_azure' not in st.session_state:
        st.session_state.use_azure = False
    if 'use_azure' not in st.session_state:
        st.session_state.use_azure = False
    if 'model_generation_model' not in st.session_state:
        st.session_state.model_generation_model = 'claude-sonnet-4-20250514'
    if 'extraction_model' not in st.session_state:
        st.session_state.extraction_model = 'gpt-4.1-2025-04-14'
    if 'use_azure' not in st.session_state:
        st.session_state.use_azure = False
    if 'model_generation_model' not in st.session_state:
        st.session_state.model_generation_model = 'claude-sonnet-4-20250514'
    if 'extraction_model' not in st.session_state:
        st.session_state.extraction_model = 'gpt-4.1-2025-04-14'
    if 'use_azure' not in st.session_state:
        st.session_state.use_azure = False

def ensure_use_cases_folder():
    """Ensure Use-cases folder exists at application start"""
    use_cases_dir = "Use-cases"
    if not os.path.exists(use_cases_dir):
        os.makedirs(use_cases_dir)
        st.success(f"✅ Created {use_cases_dir} folder")
    return use_cases_dir

def create_use_case_folder(use_case_name: str) -> str:
    """Create folder for specific use case"""
    use_cases_dir = ensure_use_cases_folder()
    # Create safe folder name
    safe_name = use_case_name.replace(' ', '_').replace('-', '_').replace('/', '_').replace('\\', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
    
    use_case_folder = os.path.join(use_cases_dir, safe_name)
    if not os.path.exists(use_case_folder):
        os.makedirs(use_case_folder)
    
    return use_case_folder

def get_use_case_path(use_case_name: str, filename: str) -> str:
    """Get full path for a file in the use case folder"""
    use_case_folder = create_use_case_folder(use_case_name)
    return os.path.join(use_case_folder, filename)

def load_extraction_context_from_current_config():
    """Load extraction context from the current configuration for the extraction phase"""
    if not st.session_state.use_case:
        return  # No use case defined yet
    
    try:
        # Get the config file path for current use case
        config_path = get_use_case_path(st.session_state.use_case, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                extraction_config = config_data.get('extraction_config', {})
                
                # Load the three extraction context fields
                st.session_state.extraction_purpose = extraction_config.get('purpose_of_extraction', '')
                
                # Handle document_type - remove the appended warning if present
                doc_type_raw = extraction_config.get('document_type', '')
                if ". Do not attempt to extract data from non-related documents" in doc_type_raw:
                    st.session_state.document_type = doc_type_raw.split(". Do not attempt to extract data from non-related documents")[0]
                else:
                    st.session_state.document_type = doc_type_raw
                
                st.session_state.custom_instructions = extraction_config.get('additional_instructions', '')
    except Exception as e:
        # If there's any error loading, just use empty defaults
        st.session_state.extraction_purpose = ""
        st.session_state.document_type = ""
        st.session_state.custom_instructions = ""

def save_extraction_context_to_config():
    """Save the current extraction context (purpose, document type, custom instructions) to config.json"""
    if not st.session_state.use_case:
        return  # No use case defined yet
    
    try:
        # Get the config file path for current use case
        config_path = get_use_case_path(st.session_state.use_case, "config.json")
        if os.path.exists(config_path):
            # Load existing config
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update the extraction context fields
            extraction_config = config_data.get('extraction_config', {})
            
            # Get session state values with fallbacks
            purpose = getattr(st.session_state, 'extraction_purpose', '')
            doc_type = getattr(st.session_state, 'document_type', '')
            custom = getattr(st.session_state, 'custom_instructions', '')
            
            extraction_config['purpose_of_extraction'] = purpose.strip() if purpose and purpose.strip() else ""
            extraction_config['document_type'] = f"{doc_type.strip()}" if doc_type and doc_type.strip() else ""
            extraction_config['additional_instructions'] = custom.strip() if custom and custom.strip() else ""
            
            # Debug: log what we're saving
            import logging
            logging.info(f"Saving extraction context: purpose='{purpose}', doc_type='{doc_type}', custom='{custom}'")
            
            # Save updated config back to file
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        # Log the error for debugging
        import logging
        logging.error(f"Error saving extraction context: {e}")
        # Don't raise the error to avoid breaking the extraction flow

def build_additional_instructions() -> str:
    """Build combined additional instructions from the three components"""
    instructions = []
    
    # Add structured purpose and document type instruction
    if st.session_state.extraction_purpose.strip() and st.session_state.document_type.strip():
        purpose_instruction = (
            f"The purpose of this extraction task is {st.session_state.extraction_purpose.strip()}. "
            f"Therefore, the document should be related to {st.session_state.document_type.strip()}. "
            f"Do not attempt to extract data from non-related documents. "
            f"If the documents are not related, output 'n/a' for all fields."
        )
        instructions.append(purpose_instruction)
    
    # Add custom instructions if provided
    if st.session_state.custom_instructions.strip():
        instructions.append(st.session_state.custom_instructions.strip())
    
    return '\n\n'.join(instructions)

def validate_azure_configuration() -> tuple[bool, str, str]:
    """Validate Azure configuration and return status, message, and CSS class"""
    use_azure = st.session_state.get('use_azure', False)
    generation_model = st.session_state.get('model_generation_model', 'claude-sonnet-4-20250514')
    extraction_model = st.session_state.get('extraction_model', 'gpt-4.1-2025-04-14')
    
    if not use_azure:
        return True, "", ""
    
    # Check if models are OpenAI
    is_generation_openai = generation_model and 'gpt' in generation_model.lower()
    is_extraction_openai = extraction_model and 'gpt' in extraction_model.lower()
    
    # Case 1: Both OpenAI models with Azure
    if is_generation_openai and is_extraction_openai:
        return True, "🔒 Using AZURE endpoint for data model generation and extraction", "certificate-green"
    
    # Case 2: Claude for generation, OpenAI for extraction with Azure
    elif not is_generation_openai and is_extraction_openai:
        return True, "🔒 Using AZURE endpoint for knowledge extraction only", "certificate-green"
    
    # Case 3: Claude for both generation and extraction with Azure (blocked)
    elif not is_generation_openai and not is_extraction_openai:
        return False, "🚫 This application does not support CLAUDE with AZURE endpoint. Select OpenAI model for extraction task to proceed with AZURE endpoint for knowledge extraction", "certificate-red"
    
    # Case 4: OpenAI for generation, Claude for extraction with Azure (blocked)
    elif is_generation_openai and not is_extraction_openai:
        return False, "🚫 This application does not support CLAUDE with AZURE endpoint. Select OpenAI model for extraction task to proceed with AZURE endpoint for knowledge extraction", "certificate-red"
    
    return True, "", ""

def parse_additional_instructions(combined_instructions: str):
    """Parse combined additional instructions back into components"""
    if not combined_instructions.strip():
        return "", "", ""
    
    # Split by double newlines to get separate instruction blocks
    instruction_blocks = combined_instructions.split('\n\n')
    
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

def load_saved_models() -> List[Dict[str, str]]:
    """Load list of saved model configurations from Use-cases folders"""
    models = []
    use_cases_dir = "Use-cases"
    
    if os.path.exists(use_cases_dir):
        for use_case_folder in os.listdir(use_cases_dir):
            folder_path = os.path.join(use_cases_dir, use_case_folder)
            if os.path.isdir(folder_path):
                # Look for config.json in each use case folder
                config_file = os.path.join(folder_path, "config.json")
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                            extraction_config = config_data.get('extraction_config', {})
                            
                            models.append({
                                'folder': use_case_folder,
                                'use_case': extraction_config.get('use_case', use_case_folder),
                                'description': extraction_config.get('description', 'No description'),
                                'model_name': extraction_config.get('main_model_name', 'Unknown'),
                                'field_count': len(extraction_config.get('fields', [])),
                                'created_at': extraction_config.get('created_at', 'Unknown')
                            })
                    except Exception as e:
                        # Skip invalid config files
                        continue
    
    return models

def save_model_config(config_data: Dict[str, Any], use_case_name: str):
    """Save model configuration to use-case folder"""
    config_path = get_use_case_path(use_case_name, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    return config_path

def load_model_config(relative_path: str) -> Dict[str, Any]:
    """Load model configuration from use-case folder"""
    full_path = os.path.join("Use-cases", relative_path)
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_field_form(field_index: int, field_data: Optional[Dict] = None):
    """Create a compact form for a single field configuration"""
    
    with st.container():
        st.markdown(f'<div class="field-container">', unsafe_allow_html=True)
        
        col_header, col_remove = st.columns([4, 1])
        with col_header:
            st.markdown(f"**📝 Field {field_index + 1}**")
        with col_remove:
            if st.button("🗑️", key=f"remove_{field_index}", help="Remove field"):
                st.session_state.fields.pop(field_index)
                st.rerun()
        
        # Four equal-width columns for the four attributes
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            field_name = st.text_input(
                "Field Name",
                value=field_data.get('field_name', '') if field_data else '',
                key=f"field_name_{field_index}",
                help="Name of the field to extract"
            )
        
        with col2:
            field_description = st.text_area(
                "Field Description",
                value=field_data.get('description', '') if field_data else '',
                key=f"description_{field_index}",
                height=80,
                help="What information to extract"
            )
        
        with col3:
            # Categories/Classes (optional)
            categories_input = st.text_area(
                "Categories/Classes (Optional)",
                value='\n'.join(field_data.get('enum_values', [])) if field_data and field_data.get('enum_values') else '',
                key=f"categories_{field_index}",
                height=80,
                help="One category per line for classification"
            )
            categories = [cat.strip() for cat in categories_input.split('\n') if cat.strip()] if categories_input.strip() else None
        
        with col4:
            # Determine if this should be enum based on categories
            available_types = ['str', 'int', 'float', 'bool', 'list[str]']
            if categories:
                available_types.extend(['enum', 'list[enum]'])
            
            current_type = field_data.get('field_type', 'str') if field_data else 'str'
            if current_type not in available_types:
                current_type = 'str'
            
            field_type = st.selectbox(
                "Data Type",
                options=available_types,
                index=available_types.index(current_type),
                key=f"field_type_{field_index}",
                help="Choose appropriate data type"
            )
        
        # Required checkbox (full width)
        required = st.checkbox(
            "Required Field",
            value=field_data.get('required', True) if field_data else True,
            key=f"required_{field_index}"
        )
        
        # Update session state
        st.session_state.fields[field_index] = {
            'field_name': field_name,
            'field_type': field_type,
            'description': field_description,
            'required': required,
            'enum_values': categories
        }
        
        st.markdown('</div>', unsafe_allow_html=True)

def validate_configuration() -> tuple[bool, List[str]]:
    """Validate the current configuration"""
    errors = []
    
    if not st.session_state.use_case.strip():
        errors.append("Use case name is required")
    
    if not st.session_state.main_model_name.strip():
        errors.append("Main model name is required")
    
    if not st.session_state.fields:
        errors.append("At least one field is required")
    
    field_names = []
    for i, field in enumerate(st.session_state.fields):
        if not field.get('field_name', '').strip():
            errors.append(f"Field {i+1}: Field name is required")
        else:
            if field['field_name'] in field_names:
                errors.append(f"Field {i+1}: Duplicate field name '{field['field_name']}'")
            field_names.append(field['field_name'])
        
        if not field.get('description', '').strip():
            errors.append(f"Field {i+1}: Description is required")
        
        if field.get('field_type') in ['enum', 'list[enum]'] and not field.get('enum_values'):
            errors.append(f"Field {i+1}: Categories are required for enum types")
    
    return len(errors) == 0, errors

def export_configuration() -> Dict[str, Any]:
    """Export the current configuration"""
    fields_config = []
    for field in st.session_state.fields:
        field_config = {
            'field_name': field['field_name'],
            'field_type': field['field_type'],
            'description': field['description'],
            'required': field['required'],
            'enum_values': field['enum_values'] if field.get('enum_values') else None
        }
        fields_config.append(field_config)
    
    config = {
        'extraction_config': {
            'use_case': st.session_state.use_case,
            'description': st.session_state.description,
            'main_model_name': st.session_state.main_model_name,
            'purpose_of_extraction': st.session_state.extraction_purpose.strip() if st.session_state.extraction_purpose.strip() else "",
            'document_type': f"{st.session_state.document_type.strip()}" if st.session_state.document_type.strip() else "",
            'additional_instructions': st.session_state.custom_instructions.strip() if st.session_state.custom_instructions.strip() else "",
            'created_at': datetime.now().isoformat(),
            'fields': fields_config
        }
    }
    
    return config

def configuration_section():
    """Configuration section of the UI"""
    st.markdown('<div class="section-header">🔧 Use Case Configuration</div>', unsafe_allow_html=True)
    
    # Ensure Use-cases folder exists
    ensure_use_cases_folder()
    
    # Model selection or creation with enhanced display
    saved_models = load_saved_models()
    
    if saved_models:
        st.markdown("**📋 Available Use Cases**")
        
        # Create selection options with detailed descriptions
        model_options = ["🆕 Create New Use Case"]
        model_display_names = {}
        
        for model in saved_models:
            display_name = f"📁 {model['use_case']} ({model['model_name']}) - {model['field_count']} fields"
            model_options.append(display_name)
            model_display_names[display_name] = model
        
        selected_option = st.selectbox(
            "Choose an extraction use case to load or create new",
            model_options,
            key="model_selection"
        )
        
        if selected_option != "🆕 Create New Use Case":
            selected_model = model_display_names[selected_option]
            
            # Show model details in an expandable section
            with st.expander(f"📋 Model Details: {selected_model['use_case']}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Use Case:** {selected_model['use_case']}")
                    st.write(f"**Model Name:** {selected_model['model_name']}")
                    st.write(f"**Fields:** {selected_model['field_count']}")
                with col2:
                    st.write(f"**Description:** {selected_model['description'][:100]}...")
                    st.write(f"**Created:** {selected_model['created_at'][:10] if selected_model['created_at'] != 'Unknown' else 'Unknown'}")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("📁 Load Selected Model", type="primary"):
                    try:
                        config_path = f"{selected_model['folder']}/config.json"
                        config_data = load_model_config(config_path)
                        extraction_config = config_data['extraction_config']
                        
                        st.session_state.use_case = extraction_config.get('use_case', '')
                        st.session_state.description = extraction_config.get('description', '')
                        st.session_state.main_model_name = extraction_config.get('main_model_name', '')
                        
                        # Handle both old format (combined additional_instructions) and new format (three separate fields)
                        if 'purpose_of_extraction' in extraction_config or 'document_type' in extraction_config:
                            # New format with separate fields
                            st.session_state.extraction_purpose = extraction_config.get('purpose_of_extraction', '')
                            # Remove the appended warning text from document_type
                            doc_type_raw = extraction_config.get('document_type', '')
                            if ". Do not attempt to extract data from non-related documents" in doc_type_raw:
                                st.session_state.document_type = doc_type_raw.split(". Do not attempt to extract data from non-related documents")[0]
                            else:
                                st.session_state.document_type = doc_type_raw
                            st.session_state.custom_instructions = extraction_config.get('additional_instructions', '')
                        else:
                            # Old format with combined additional_instructions - parse it
                            combined_instructions = extraction_config.get('additional_instructions', '')
                            purpose, doc_type, custom = parse_additional_instructions(combined_instructions)
                            st.session_state.extraction_purpose = purpose
                            st.session_state.document_type = doc_type
                            st.session_state.custom_instructions = custom
                        
                        # Build combined instructions for backward compatibility
                        st.session_state.additional_instructions = build_additional_instructions()
                        
                        st.session_state.fields = extraction_config.get('fields', [])
                        
                        st.success(f"✅ Loaded model: {selected_model['use_case']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error loading model: {str(e)}")
            with col2:
                if st.button("🗑️ Delete Model"):
                    try:
                        folder_path = f"Use-cases/{selected_model['folder']}"
                        import shutil
                        shutil.rmtree(folder_path)
                        st.success(f"✅ Deleted model: {selected_model['use_case']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error deleting model: {str(e)}")
        
        st.markdown("---")
    else:
        st.info("💡 No existing models found. Create your first model below.")
    
    # Basic configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.use_case = st.text_input(
            "Use Case Name",
            value=st.session_state.use_case,
            help="Name for your extraction use case"
        )
        
        st.session_state.main_model_name = st.text_input(
            "Model Name",
            value=st.session_state.main_model_name,
            help="Name for the generated model class"
        )
    
    with col2:
        st.session_state.description = st.text_area(
            "Description",
            value=st.session_state.description,
            height=100,
            help="What you're extracting and why"
        )
        
        if st.button("🔤 Auto-Generate Model Name"):
            if st.session_state.use_case:
                suggested = st.session_state.use_case.replace(' ', '').replace('-', '').replace('_', '')
                if not suggested.endswith('Info'):
                    suggested += 'Info'
                st.session_state.main_model_name = suggested
                st.rerun()
    
    # Model Selection Settings
    st.markdown("**⚙️ Model Settings**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Initialize session state for model settings if not exists
        if 'model_generation_model' not in st.session_state:
            st.session_state.model_generation_model = 'claude-sonnet-4-20250514'
        if 'extraction_model' not in st.session_state:
            st.session_state.extraction_model = 'gpt-4.1-2025-04-14'
        if 'use_azure' not in st.session_state:
            st.session_state.use_azure = False
        
        st.session_state.model_generation_model = st.selectbox(
            "Model for Data Model Generation",
            options=['claude-sonnet-4-20250514', 'gpt-4.1-2025-04-14'],
            index=0 if st.session_state.model_generation_model == 'claude-sonnet-4-20250514' else 1,
            help="Choose the AI model to generate Pydantic data models from your field configurations"
        )
    
    with col2:
        # Extraction model options depend on Azure selection
        azure_enabled = st.session_state.get('use_azure', False)
        if azure_enabled:
            extraction_options = ['gpt-4.1-2025-04-14']
            default_extraction = 'gpt-4.1-2025-04-14'
        else:
            extraction_options = ['gpt-4.1-2025-04-14', 'claude-sonnet-4-20250514']
            default_extraction = st.session_state.extraction_model
            
        if default_extraction not in extraction_options:
            default_extraction = extraction_options[0]
            
        st.session_state.extraction_model = st.selectbox(
            "Extraction Model",
            options=extraction_options,
            index=extraction_options.index(default_extraction),
            help="Choose the AI model for extracting data from documents",
            disabled=azure_enabled and st.session_state.extraction_model != 'gpt-4.1-2025-04-14'
        )
        
        # Show disabled message if Azure is enabled and Claude was selected
        if azure_enabled and st.session_state.extraction_model and 'claude' in st.session_state.extraction_model.lower():
            st.warning("⚠️ This application uses only OpenAI model with Azure endpoint for extraction task")
    
    with col3:
        st.session_state.use_azure = st.checkbox(
            "Use Microsoft Azure Endpoint",
            value=st.session_state.get('use_azure', False),
            help="Select for secure data processing. Requires MS AZURE API key"
        )
        
        # Force GPT-4.1 selection when Azure is enabled
        if st.session_state.use_azure and st.session_state.extraction_model != 'gpt-4.1-2025-04-14':
            st.session_state.extraction_model = 'gpt-4.1-2025-04-14'
            st.rerun()
    
    # Azure Configuration Validation and Certificate Display
    is_valid, message, css_class = validate_azure_configuration()
    if message:
        st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)
    
    # Fields configuration
    st.markdown("**📋 Extraction Fields**")
    
    col_add, col_info = st.columns([1, 4])
    with col_add:
        if st.button("➕ Add Field", type="primary", key="add_field_btn"):
            new_field = {
                'field_name': '',
                'field_type': 'str',
                'description': '',
                'required': True,
                'enum_values': None
            }
            st.session_state.fields.append(new_field)
            st.rerun()
    
    with col_info:
        if st.session_state.fields:
            st.markdown(f"*Currently configured: {len(st.session_state.fields)} fields*")
        else:
            st.markdown("*No fields defined yet*")
    
    # Display fields
    if st.session_state.fields:
        for i, field in enumerate(st.session_state.fields):
            create_field_form(i, field)
    
    # Validation and save
    is_valid, errors = validate_configuration()
    
    if errors:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**⚠️ Configuration Issues:**")
        for error in errors:
            st.markdown(f"• {error}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if is_valid:
        st.markdown('<div class="success-box">✅ Configuration is ready!</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Model", type="primary"):
                try:
                    config = export_configuration()
                    config_path = save_model_config(config, st.session_state.use_case)
                    use_case_folder = os.path.dirname(config_path)
                    st.success(f"✅ Model saved to: {use_case_folder}")
                except Exception as e:
                    st.error(f"❌ Error saving: {str(e)}")
        
        with col2:
            config = export_configuration()
            config_json = json.dumps(config, indent=2)
            st.download_button(
                label="📥 Download",
                data=config_json,
                file_name=f"{st.session_state.use_case.replace(' ', '_').lower()}_config.json",
                mime="application/json"
            )
        
        with col3:
            if st.button("🚀 Go to Extraction", type="secondary"):
                load_extraction_context_from_current_config()  # Load context before switching
                st.session_state.current_tab = "Extraction"
                st.rerun()

def extraction_section():
    """Extraction section of the UI"""
    st.markdown('<div class="section-header">🎯 Data Extraction</div>', unsafe_allow_html=True)
    
    # Check if configuration is ready
    is_valid, _ = validate_configuration()
    
    if not is_valid:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**⚠️ Please complete model configuration first**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("← Back to Configuration"):
            st.session_state.current_tab = "Configuration"
            st.rerun()
        return
    
    # Extraction context and instructions
    st.markdown("**📋 Extraction Context**")
    st.info("💡 **Enhance your extraction quality!** Providing context about your extraction purpose and document types helps the AI understand your specific needs and deliver more accurate results.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.extraction_purpose = st.text_input(
            "Purpose of Extraction",
            value=st.session_state.extraction_purpose,
            help="e.g., to extract the information against key fields from AI consultancy document(s)",
            placeholder="e.g., to extract information from AI consultancy reports"
        )
    
    with col2:
        st.session_state.document_type = st.text_input(
            "Type of Document(s)",
            value=st.session_state.document_type,
            help="e.g., Documents for AI consultancy to companies",
            placeholder="e.g., AI consultancy reports, business documents"
        )
    
    # Fields are now optional - no validation errors needed
    
    # Additional custom instructions (optional)
    default_custom_instructions = ""
    if 'custom_instructions' not in st.session_state or not st.session_state.custom_instructions:
        st.session_state.custom_instructions = default_custom_instructions
    
    st.session_state.custom_instructions = st.text_area(
        "Additional Custom Instructions (Optional)",
        value=st.session_state.custom_instructions,
        help="Additional specific instructions for the extraction process",
        placeholder="e.g., Focus on the first page only, ignore footnotes, use specific date formats"
    )
    
    # File selection
    st.markdown("**📁 Document Selection**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selection_mode = st.radio(
            "Selection Mode",
            ["Select Folder", "Select Individual Files"],
            key="selection_mode"
        )
    
    with col2:
        if selection_mode == "Select Folder":
            col_input, col_browse = st.columns([3, 1])
            
            # Initialize folder path in session state if not exists
            if 'selected_folder_path' not in st.session_state:
                st.session_state.selected_folder_path = ""
            
            with col_input:
                folder_path = st.text_input(
                    "Folder Path",
                    value=st.session_state.selected_folder_path,
                    help="Path to folder containing documents",
                    key="folder_path_input"
                )
                # Update session state when user types
                if folder_path != st.session_state.selected_folder_path:
                    st.session_state.selected_folder_path = folder_path
            
            with col_browse:
                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                if st.button("📁 Browse", help="Open folder selection dialog"):
                    # Use tkinter for folder selection dialog
                    try:
                        import tkinter as tk
                        from tkinter import filedialog
                        
                        # Create a root window and hide it
                        root = tk.Tk()
                        root.withdraw()
                        root.attributes('-topmost', True)
                        
                        # Open folder dialog
                        selected_folder = filedialog.askdirectory(
                            title="Select Folder Containing Documents"
                        )
                        
                        # Destroy the root window
                        root.destroy()
                        
                        if selected_folder:
                            st.session_state.selected_folder_path = selected_folder
                            st.rerun()
                    except ImportError:
                        st.error("❌ Folder dialog not available. Please enter path manually.")
                    except Exception as e:
                        st.error(f"❌ Error opening folder dialog: {str(e)}")
            
            # Use the session state value for folder processing
            folder_path = st.session_state.selected_folder_path
            
            if folder_path and os.path.exists(folder_path):
                supported_files = []
                for ext in ['.pdf', '.docx', '.doc']:
                    supported_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
                
                if supported_files:
                    st.success(f"✅ Found {len(supported_files)} supported documents")
                    st.session_state.selected_files = supported_files
                    
                    # Show preview of found files
                    with st.expander(f"📋 Preview Files ({len(supported_files)} files)"):
                        for file_path in supported_files[:10]:  # Show first 10
                            st.text(f"📄 {os.path.basename(file_path)}")
                        if len(supported_files) > 10:
                            st.text(f"... and {len(supported_files) - 10} more files")
                else:
                    st.warning("No supported documents (.pdf, .docx, .doc) found in folder")
            elif folder_path:
                st.error("❌ Folder path does not exist")
        else:
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=['pdf', 'docx', 'doc'],
                accept_multiple_files=True,
                help="Select one or more documents to process"
            )
            
            if uploaded_files:
                # Save uploaded files temporarily
                temp_files = []
                for uploaded_file in uploaded_files:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_files.append(temp_path)
                
                st.session_state.selected_files = temp_files
                st.success(f"✅ Selected {len(temp_files)} documents")
    
    # Clear files button
    if st.session_state.selected_files:
        if st.button("🗑️ Clear Files", help="Clear selected files and clean up temporary files"):
            _cleanup_temp_files()
            st.success("✅ Files cleared and temporary files cleaned up")
            st.rerun()
    
    # Extraction execution
    if st.session_state.selected_files:
        st.markdown("**⚡ Run Extraction**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Documents", len(st.session_state.selected_files))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Fields", len(st.session_state.fields))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Model", st.session_state.main_model_name or "Not set")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Rebuild models checkbox
        st.session_state.rebuild_models = st.checkbox(
            "🔧 Rebuild/Rebuild models", 
            value=st.session_state.rebuild_models,
            help="Rebuild models for a new use case, or rebuild existing model only when you have edited the existing model or modified additional instructions"
        )
        
        # Check Azure configuration validity before allowing extraction
        azure_valid, azure_message, azure_css = validate_azure_configuration()
        
        if st.button("🚀 Start Extraction", type="primary", use_container_width=True, disabled=not azure_valid):
            if azure_valid:
                run_extraction()
            else:
                st.error("❌ Cannot start extraction due to invalid Azure configuration. Please check your model settings.")
    
    # Display results
    if st.session_state.extraction_results is not None:
        display_results()

def run_extraction():
    """Execute the extraction process"""
    try:
        # Get API configuration and model selections
        api_config = get_api_config()
        model_generation_model = st.session_state.get('model_generation_model', 'claude-sonnet-4-20250514')
        extraction_model = st.session_state.get('extraction_model', 'gpt-4.1-2025-04-14')
        
        # Ensure models are strings and not None
        if not model_generation_model:
            model_generation_model = 'claude-sonnet-4-20250514'
        if not extraction_model:
            extraction_model = 'gpt-4.1-2025-04-14'
        
        # Initialize components with model selections and API config
        # Pass API config for OpenAI model generation when Azure is enabled
        if model_generation_model and 'gpt' in model_generation_model.lower() and api_config.get('use_azure', False):
            model_generator = ModelGenerator(model_selection=model_generation_model, api_config=api_config)
        else:
            model_generator = ModelGenerator(model_selection=model_generation_model)
        document_parser = DocumentParser()
        
        # Initialize extractor based on extraction model selection
        if extraction_model and 'claude' in extraction_model.lower():
            # If Claude is selected for extraction (only when not using Azure)
            extractor = ClaudeExtractor(model_selection=extraction_model)
        else:
            # Use OpenAI for extraction
            extractor = OpenAIExtractor(api_config=api_config)
        
        # Generate models and prompts
        config = export_configuration()
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Handle models (generate or load existing)
        use_case_name = st.session_state.use_case
        safe_use_case = use_case_name.replace(' ', '_').replace('-', '_')
        model_filename = f"{safe_use_case}_models.py"
        prompt_filename = f"{safe_use_case}_prompt.py"
        model_path = get_use_case_path(use_case_name, model_filename)
        prompt_path = get_use_case_path(use_case_name, prompt_filename)
        
        if st.session_state.rebuild_models:
            # Generate new models
            model_display_name = "Claude" if model_generation_model and 'claude' in model_generation_model.lower() else "OpenAI GPT-4.1"
            status_text.text(f"🔧 Generating new Pydantic models using {model_display_name}...")
            progress_bar.progress(10)
            try:
                pydantic_model_class, model_code = model_generator.generate_models_from_config_data(config)
                progress_bar.progress(20)
                
                status_text.text("💾 Saving generated models and prompts...")
                extraction_prompt = model_generator.get_extraction_prompt()
                
                # Save Python models and prompt to use-case folder
                model_generator.save_generated_models(model_path, model_code)
                model_generator.save_extraction_prompt(prompt_path)
                progress_bar.progress(30)
                
                status_text.text("✅ Models generated and saved successfully")
            except Exception as model_error:
                progress_bar.progress(0)
                status_text.text("❌ Model generation failed")
                st.error(f"❌ Model generation failed: {str(model_error)}")
                # Show the problematic configuration for debugging
                with st.expander("🔍 Debug Information"):
                    st.json(config)
                    st.text("If you see this error, try simplifying your field names and descriptions.")
                return
        else:
            # Load existing models
            status_text.text("📂 Loading existing models from saved files...")
            progress_bar.progress(10)
            try:
                if not os.path.exists(model_path) or not os.path.exists(prompt_path):
                    progress_bar.progress(0)
                    status_text.text("❌ Models not found")
                    st.error(f"❌ Models not found for use case '{use_case_name}'. Please check 'Build/Rebuild models' to generate them first.")
                    return
                
                # Load existing models and prompt
                pydantic_model_class, extraction_prompt = model_generator.load_models_and_prompt(model_path, prompt_path)
                progress_bar.progress(30)
                
                status_text.text("✅ Existing models loaded successfully")
            except Exception as load_error:
                progress_bar.progress(0)
                status_text.text("❌ Failed to load existing models")
                st.error(f"❌ Failed to load existing models: {str(load_error)}")
                st.info("💡 Try checking 'Rebuild models' to generate new models")
                return
        
        # Step 2: Parse documents
        status_text.text("📄 Parsing documents...")
        progress_bar.progress(40)
        parsed_documents = []
        total_files = len(st.session_state.selected_files)
        
        for i, file_path in enumerate(st.session_state.selected_files):
            try:
                status_text.text(f"📄 Parsing document {i+1}/{total_files}: {os.path.basename(file_path)}")
                parsed_doc = document_parser.parse_document(file_path)
                parsed_documents.append(parsed_doc)
                # Update progress for parsing (40-60%)
                progress_bar.progress(40 + int(20 * (i+1) / total_files))
            except Exception as e:
                st.warning(f"⚠️ Could not parse {os.path.basename(file_path)}: {str(e)}")
        
        if not parsed_documents:
            progress_bar.progress(0)
            status_text.text("❌ No documents could be parsed")
            st.error("❌ No documents could be parsed")
            return
        
        status_text.text(f"✅ Parsed {len(parsed_documents)} documents successfully")
        progress_bar.progress(60)
        
        # Step 3: Extract data
        # Dynamic extraction model display
        extraction_display_name = "Claude" if extraction_model and 'claude' in extraction_model.lower() else "OpenAI GPT-4.1"
        azure_text = " (Azure)" if api_config.get('use_azure', False) else ""
        status_text.text(f"🧠 Extracting data from {len(parsed_documents)} documents using {extraction_display_name}{azure_text}...")
        progress_bar.progress(70)
        
        results = extractor.extract_batch(
            documents=parsed_documents,
            extraction_prompt=extraction_prompt,
            model_class=pydantic_model_class,
            additional_instructions=build_additional_instructions()
        )
        
        progress_bar.progress(90)
        status_text.text("✅ Data extraction completed")
        
        # Save results 
        status_text.text("💾 Saving extraction results...")
        st.session_state.extraction_results = results
        
        # Save results to use-case folder
        if results:
            try:
                clean_data = []
                for result in results:
                    clean_result = {k: v for k, v in result.items() if not k.startswith('_')}
                    for key, value in clean_result.items():
                        if hasattr(value, 'value'):
                            clean_result[key] = value.value
                        elif isinstance(value, list):
                            if value and hasattr(value[0], 'value'):
                                clean_result[key] = '; '.join([item.value for item in value])
                            else:
                                clean_result[key] = '; '.join([str(item) for item in value])
                    clean_data.append(clean_result)
                
                df = pd.DataFrame(clean_data)
                results_filename = f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                results_path = get_use_case_path(use_case_name, results_filename)
                df.to_excel(results_path, index=False, engine='openpyxl')
                
                st.info(f"📁 Results also saved to: {results_path}")
            except Exception as save_error:
                st.warning(f"⚠️ Could not save results to use-case folder: {str(save_error)}")
        
        # Clean up temporary files
        _cleanup_temp_files()
        
        # Save extraction context to config.json after successful extraction
        save_extraction_context_to_config()
        
        # Complete progress
        progress_bar.progress(100)
        status_text.text("🎉 Extraction completed successfully!")
        
        # Clear progress bar after a short delay
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Extraction completed! Processed {len(results)} documents")
        st.rerun()
        
    except Exception as e:
        # Clean up temporary files even if extraction fails
        _cleanup_temp_files()
        
        st.error(f"❌ Extraction failed: {str(e)}")
        
        # Show debug information
        with st.expander("🔍 Debug Information"):
            st.text("Error details:")
            st.text(str(e))
            if 'config' in locals():
                st.text("Configuration:")
                st.json(config)
            
            st.text("Troubleshooting tips:")
            st.text("1. Check that your API keys are correctly set in .env file")
            st.text("2. Ensure field names contain only letters, numbers, and spaces")
            st.text("3. Keep field descriptions concise and clear")
            st.text("4. For enum fields, make sure categories are provided")

def _cleanup_temp_files():
    """Clean up temporary files created during document upload"""
    import os
    import glob
    
    try:
        # Clean up temp files from uploaded documents
        if hasattr(st.session_state, 'selected_files') and st.session_state.selected_files:
            for file_path in st.session_state.selected_files:
                if file_path and file_path.startswith("temp_") and os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                        logger.info(f"Cleaned up temp file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not clean up temp file {file_path}: {e}")
        
        # Also clean up any orphaned temp_ files in the project root
        temp_files = glob.glob("temp_*")
        for temp_file in temp_files:
            try:
                if os.path.isfile(temp_file):
                    os.unlink(temp_file)
                    logger.info(f"Cleaned up orphaned temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Could not clean up orphaned temp file {temp_file}: {e}")
                
        # Clear the selected files from session state
        st.session_state.selected_files = []
        
    except Exception as e:
        logger.error(f"Error during temp file cleanup: {e}")

def display_results():
    """Display extraction results in a well-formatted box"""
    results = st.session_state.extraction_results
    
    if not results:
        st.warning("No results to display")
        return
    
    # Create the results container
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown('<div class="results-header">📊 Extraction Results</div>', unsafe_allow_html=True)
    
    # Results summary
    st.markdown('<div class="result-summary">', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Documents Processed", len(results))
    
    with col2:
        successful_extractions = len([r for r in results if not r.get('_document_metadata', {}).get('extraction_error')])
        st.metric("✅ Successful", successful_extractions)
    
    with col3:
        failed_extractions = len(results) - successful_extractions
        st.metric("❌ Failed", failed_extractions)
    
    with col4:
        if results:
            first_result = {k: v for k, v in results[0].items() if not k.startswith('_')}
            st.metric("📊 Fields Extracted", len(first_result))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create clean DataFrame for display (only extracted fields)
    clean_data = []
    for i, result in enumerate(results):
        clean_result = {}
        
        # Process extracted fields only
        for key, value in result.items():
            if not key.startswith('_'):
                if hasattr(value, 'value'):  # Enum object
                    clean_result[key] = value.value
                elif isinstance(value, list):
                    if value and hasattr(value[0], 'value'):  # List of enums
                        clean_result[key] = '; '.join([item.value for item in value])
                    else:  # List of strings
                        clean_result[key] = '; '.join([str(item) for item in value])
                else:
                    clean_result[key] = str(value) if value is not None else 'n/a'
        
        clean_data.append(clean_result)
    
    df = pd.DataFrame(clean_data)
    
    # Display the data table with enhanced formatting
    st.markdown("### 📋 Extracted Data")
    
    # Show extracted data table
    st.dataframe(
        df, 
        use_container_width=True,
        hide_index=True
    )
    
    if len(results) > 10:
        st.info(f"Showing all {len(results)} documents processed.")
    
    # Download section
    st.markdown('<div class="download-container">', unsafe_allow_html=True)
    st.markdown("### 💾 Download Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # CSV download
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📥 CSV Format",
            data=csv_data,
            file_name=f"{st.session_state.use_case.replace(' ', '_').lower()}_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel download
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        
        st.download_button(
            "📊 Excel Format",
            data=excel_buffer.getvalue(),
            file_name=f"{st.session_state.use_case.replace(' ', '_').lower()}_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        # JSON download
        json_data = json.dumps([{k: v for k, v in result.items() if not k.startswith('_')} for result in results], indent=2, default=str)
        st.download_button(
            "📋 JSON Format",
            data=json_data,
            file_name=f"{st.session_state.use_case.replace(' ', '_').lower()}_results.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col4:
        # Clear results
        if st.button("🔄 New Extraction", use_container_width=True):
            st.session_state.extraction_results = None
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main UI function"""
    initialize_session_state()
    
    # Ensure Use-cases folder exists at startup
    ensure_use_cases_folder()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Knowledge Extraction Agent</h1>', unsafe_allow_html=True)
    
    # Navigation tabs
    col1, col2 = st.columns(2)
    with col1:
        config_style = "primary" if st.session_state.current_tab == "Configuration" else "secondary"
        if st.button("⚙️ Configuration", type=config_style, use_container_width=True):
            st.session_state.current_tab = "Configuration"
            st.rerun()
    
    with col2:
        extraction_style = "primary" if st.session_state.current_tab == "Extraction" else "secondary"
        if st.button("🎯 Extraction", type=extraction_style, use_container_width=True):
            load_extraction_context_from_current_config()  # Load context before switching
            st.session_state.current_tab = "Extraction"
            st.rerun()
    
    st.markdown("---")
    
    # Session state-based tab navigation
    if st.session_state.current_tab == "Configuration":
        configuration_section()
    elif st.session_state.current_tab == "Extraction":
        extraction_section()

if __name__ == "__main__":
    # Add required import for Excel download
    from io import BytesIO

    main()

