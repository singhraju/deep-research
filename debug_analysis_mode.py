#!/usr/bin/env python3
"""
Debug script to test analysis mode resolution and LLM prompt/response.
This helps diagnose why no analysis_mode is selected from questions.
"""

import json
import logging
import sys
import os
from pathlib import Path

# Add the packages to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "packages" / "agents" / "src"))
sys.path.append(str(project_root / "packages" / "core" / "src"))
sys.path.append(str(project_root / "packages" / "utils" / "src"))

from deep_research_agents.user_intent import (
    build_app,
    load_semantic_yaml,
    extract_analysis_modes,
    build_analysis_mode_index,
    resolve_analysis_mode_with_llm,
    build_llm,
    ANALYSIS_MODE_SELECTION_SYSTEM_PROMPT,
    _serialize_for_llm
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_analysis_mode_resolution():
    """Test the analysis mode resolution process step by step."""
    
    print("=" * 80)
    print("DEBUG: Analysis Mode Resolution Test")
    print("=" * 80)
    
    # Load semantic model
    semantic_model_path = "configs/correlation_pattern/old_config/ecap_semantic_view_with_samples.yaml"
    semantic_model = load_semantic_yaml(semantic_model_path)
    print(f"✅ Loaded semantic model from: {semantic_model_path}")
    
    # Extract analysis modes
    analysis_modes = extract_analysis_modes(semantic_model)
    print(f"📊 Found {len(analysis_modes)} analysis modes:")
    for mode in analysis_modes:
        print(f"  - {mode.get('name')}: {mode.get('description', '')[:100]}...")
    
    # Build index
    analysis_mode_index = build_analysis_mode_index(analysis_modes)
    print(f"🔍 Built analysis mode index with {len(analysis_mode_index)} entries")
    
    # Test question that should match the cost change mode
    test_question = "Find what changes in Virginia?"
    filters = [{"field": "service_area_state", "operator": "=", "value": "VA", "source": "dimension_match"}]
    
    print(f"\n🔤 Test Question: {test_question}")
    print(f"🎯 Test Filters: {filters}")
    
    # Build LLM context
    llm_context = {
        "analysis_modes": analysis_modes,
        "context_filters": {},
        "analysis_hints": {},
        "extracted_filters": filters,
    }
    
    print(f"\n📝 LLM Context:")
    context_json = _serialize_for_llm(llm_context, max_chars=2000)
    print(context_json[:1000] + "..." if len(context_json) > 1000 else context_json)
    
    # Build prompt
    user_prompt = (
        f"Question: {test_question}\n\n"
        f"Analysis mode context JSON: {_serialize_for_llm(llm_context, max_chars=12_000)}"
    )
    
    messages = [
        {"role": "system", "content": ANALYSIS_MODE_SELECTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    print(f"\n💬 System Prompt Preview:")
    print(ANALYSIS_MODE_SELECTION_SYSTEM_PROMPT[:300] + "...")
    
    print(f"\n💬 User Prompt Preview:")
    print(user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt)
    
    # Test LLM availability
    try:
        print(f"\n🤖 Testing LLM availability...")
        llm_factory = build_llm()
        llm = llm_factory()
        print(f"✅ LLM client created successfully: {type(llm).__name__}")
        
        # Test structured output
        from deep_research_agents.user_intent import AnalysisModeSelectionSchema
        structured_llm = llm.with_structured_output(AnalysisModeSelectionSchema)
        print(f"✅ Structured LLM created successfully")
        
        # Make the actual LLM call
        print(f"\n🚀 Making LLM call...")
        result_schema = structured_llm.invoke(messages)
        
        print(f"📥 LLM Response:")
        print(f"  - analysis_mode: {result_schema.analysis_mode}")
        print(f"  - type: {type(result_schema.analysis_mode)}")
        
        if result_schema.analysis_mode:
            print(f"✅ SUCCESS: LLM selected analysis mode: {result_schema.analysis_mode}")
        else:
            print(f"❌ ISSUE: LLM returned None for analysis_mode")
            print(f"🔍 This explains the 'Analysis mode missing' warning!")
            
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        logger.exception("LLM call failed")
        
    print(f"\n" + "=" * 80)

def test_full_user_intent():
    """Test the full user intent resolution flow."""
    
    print("=" * 80)
    print("DEBUG: Full User Intent Test")
    print("=" * 80)
    
    try:
        # Build the app
        yaml_path = "configs/correlation_pattern/old_config/ecap_semantic_view_with_samples.yaml"
        app = build_app(yaml_path)
        print(f"✅ Built user intent app from: {yaml_path}")
        
        # Test question
        test_question = "Find what do in Virginia?"
        context = {"hcc_medium": "IP BH"}
        
        print(f"\n🔤 Test Question: {test_question}")
        print(f"🎯 Test Context: {context}")
        
        # Run resolution
        print(f"\n🚀 Running full intent resolution...")
        result = app(test_question, context=context)
        
        print(f"\n📥 Resolution Result:")
        print(f"  - analysis_mode: {result.get('analysis_mode')}")
        print(f"  - analysis_mode_parameters: {bool(result.get('analysis_mode_parameters'))}")
        print(f"  - filters: {len(result.get('filters', []))} filters")
        print(f"  - group_by: {result.get('group_by', [])}")
        print(f"  - validation_warnings: {result.get('validation_warnings', [])}")
        
        if result.get('analysis_mode'):
            print(f"✅ SUCCESS: Full resolution selected analysis mode!")
        else:
            print(f"❌ ISSUE: Full resolution did not select analysis mode")
            print(f"🔍 This would trigger the orchestrator warning!")
            
    except Exception as e:
        print(f"❌ Full Intent Error: {e}")
        logger.exception("Full intent resolution failed")
        
    print(f"\n" + "=" * 80)

if __name__ == "__main__":
    # Set working directory to project root
    os.chdir(Path(__file__).parent)
    
    print("🐛 Starting Analysis Mode Debug Session...")
    
    # Test individual components
    test_analysis_mode_resolution()
    
    # Test full flow
    test_full_user_intent()
    
    print("🏁 Debug session complete!")
