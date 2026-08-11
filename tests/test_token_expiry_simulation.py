"""
Test script to simulate token expiry scenario without Redis.

This script simulates what happens when:
1. Agent is created with a token
2. Token expires/becomes invalid after some time
3. LLM call fails with 401
4. Retry mechanism should kick in

Run this locally to verify the fix works before deploying to server.
"""

import time
import os
import sys
from unittest.mock import patch, MagicMock
from langchain_openai import ChatOpenAI

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../packages', 'utils', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../packages', 'core', 'src'))

# Set environment to avoid Redis
os.environ['REDIS_ENABLED'] = 'false'

from deep_research_utils import EHAPBase
from deep_research_utils.ehap_retry import llm_invoke, structured_llm_invoke
from openai import AuthenticationError  # Correct import - from openai, not langchain_core


def test_token_expiry_with_retry():
    """
    Simulate token expiry and verify retry mechanism works.
    """
    print("=" * 80)
    print("TEST: Token Expiry Simulation (Without Redis)")
    print("=" * 80)
    
    # Track token generations
    token_generation_count = 0
    
    def mock_fetch_token():
        """Mock token fetch - returns different tokens each time."""
        nonlocal token_generation_count
        token_generation_count += 1
        token = f"mock_token_{token_generation_count}"
        print(f"\n🔑 Token Generated: {token} (Generation #{token_generation_count})")
        return token
    
    # Create EHAP instance
    ehap = EHAPBase()
    
    # Mock the token fetch
    with patch.object(ehap, '_fetch_new_token', side_effect=mock_fetch_token):
        
        # Step 1: Create initial LLM with first token
        print("\n" + "=" * 80)
        print("STEP 1: Create LLM with initial token")
        print("=" * 80)
        
        initial_token = ehap.get_token()
        print(f"✅ Initial token obtained: {initial_token}")
        
        llm = ChatOpenAI(
            api_key=initial_token,
            base_url="http://mock-api",
            model="gpt-4"
        )
        print(f"✅ LLM created with token: {initial_token}")
        
        # Step 2: Simulate successful call
        print("\n" + "=" * 80)
        print("STEP 2: First LLM call (should succeed)")
        print("=" * 80)
        
        mock_response = MagicMock()
        mock_response.content = "First successful response"
        
        # Patch at class level, not instance level (Pydantic models don't allow instance patching)
        with patch('langchain_openai.ChatOpenAI.invoke', return_value=mock_response):
            messages = [{"role": "user", "content": "Test message"}]
            response, updated_llm = llm_invoke(
                llm=llm,
                ehap=ehap,
                messages=messages,
                llm_reinitializer=lambda: ChatOpenAI(
                    api_key=ehap.get_token(),
                    base_url="http://mock-api",
                    model="gpt-4"
                )
            )
            print(f"✅ Response: {response.content}")
            print(f"✅ LLM still using token: {initial_token}")
        
        # Step 3: Simulate token expiry (401 error)
        print("\n" + "=" * 80)
        print("STEP 3: Simulate token expiry (401 error)")
        print("=" * 80)
        print("⏰ Simulating: 2 hours passed, token expired on EHAP side...")
        
        call_count = 0
        
        def mock_invoke_with_401(messages, **kwargs):
            """First call fails with 401, second call succeeds."""
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                print(f"❌ Call #{call_count}: 401 AuthenticationError (old token: {initial_token})")
                # Create proper AuthenticationError with required arguments
                mock_response = MagicMock()
                mock_response.status_code = 401
                raise AuthenticationError(
                    "Invalid authentication credentials",
                    response=mock_response,
                    body={"error": {"message": "Invalid authentication credentials"}}
                )
            else:
                print(f"✅ Call #{call_count}: Success with fresh token!")
                mock_resp = MagicMock()
                mock_resp.content = "Success after token refresh"
                return mock_resp
        
        # Patch at class level to simulate 401 then success
        with patch('langchain_openai.ChatOpenAI.invoke', side_effect=mock_invoke_with_401):
            print("\n🔄 Attempting LLM call with expired token...")
            
            try:
                response, updated_llm = llm_invoke(
                    llm=llm,
                    ehap=ehap,
                    messages=messages,
                    llm_reinitializer=lambda: ChatOpenAI(
                        api_key=ehap.get_token(),
                        base_url="http://mock-api",
                        model="gpt-4"
                    )
                )
                
                print("\n" + "=" * 80)
                print("RESULT: Token Refresh Successful!")
                print("=" * 80)
                print(f"✅ Response: {response.content}")
                print(f"✅ Total LLM invocations: {call_count}")
                print(f"✅ Total tokens generated: {token_generation_count}")
                print(f"✅ LLM was recreated with fresh token")
                
                # Verify the fix worked
                assert call_count == 2, f"Expected 2 calls (1 fail + 1 retry), got {call_count}"
                # Note: Token count can be 2-4 depending on pre-invocation checks and reinitializer calls
                # What matters is that the retry succeeded
                assert token_generation_count >= 2, f"Expected at least 2 tokens, got {token_generation_count}"
                print(f"   Note: {token_generation_count} tokens generated (includes pre-checks and reinitializer calls)")
                
                print("\n" + "=" * 80)
                print("✅ TEST PASSED: Token expiry handled correctly!")
                print("=" * 80)
                
            except Exception as e:
                print("\n" + "=" * 80)
                print("❌ TEST FAILED!")
                print("=" * 80)
                print(f"Error: {e}")
                raise


def test_structured_output_with_expiry():
    """
    Test structured output with token expiry.
    """
    print("\n\n" + "=" * 80)
    print("TEST: Structured Output with Token Expiry")
    print("=" * 80)
    
    from pydantic import BaseModel
    
    class TestSchema(BaseModel):
        message: str
        status: str
    
    token_generation_count = 0
    
    def mock_fetch_token():
        nonlocal token_generation_count
        token_generation_count += 1
        token = f"mock_token_{token_generation_count}"
        print(f"\n🔑 Token Generated: {token}")
        return token
    
    ehap = EHAPBase()
    
    with patch.object(ehap, '_fetch_new_token', side_effect=mock_fetch_token):
        initial_token = ehap.get_token()
        
        llm = ChatOpenAI(
            api_key=initial_token,
            base_url="http://mock-api",
            model="gpt-4"
        )
        
        call_count = 0
        
        def mock_structured_invoke(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                print(f"❌ Call #{call_count}: 401 AuthenticationError")
                # Create proper AuthenticationError with required arguments
                mock_response = MagicMock()
                mock_response.status_code = 401
                raise AuthenticationError(
                    "Invalid authentication credentials",
                    response=mock_response,
                    body={"error": {"message": "Invalid authentication credentials"}}
                )
            else:
                print(f"✅ Call #{call_count}: Success with fresh token!")
                return TestSchema(message="Success", status="ok")
        
        # Mock the with_structured_output chain
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke = mock_structured_invoke
        
        # Patch at class level
        with patch('langchain_openai.ChatOpenAI.with_structured_output', return_value=mock_structured_llm):
            messages = [{"role": "user", "content": "Test"}]
            
            result, updated_llm = structured_llm_invoke(
                llm=llm,
                ehap=ehap,
                messages=messages,
                schema=TestSchema,
                llm_reinitializer=lambda: ChatOpenAI(
                    api_key=ehap.get_token(),
                    base_url="http://mock-api",
                    model="gpt-4"
                )
            )
            
            print("\n" + "=" * 80)
            print("✅ TEST PASSED: Structured output with token refresh works!")
            print("=" * 80)
            print(f"Result: {result}")
            print(f"Total LLM calls: {call_count}")
            print(f"Total tokens generated: {token_generation_count}")
            assert call_count == 2, f"Expected 2 LLM calls, got {call_count}"
            assert token_generation_count >= 2, f"Expected at least 2 tokens, got {token_generation_count}"


def test_without_reinitializer():
    """
    Test behavior when no reinitializer is provided (should fail).
    """
    print("\n\n" + "=" * 80)
    print("TEST: Without Reinitializer (Should Fail)")
    print("=" * 80)
    
    ehap = EHAPBase()
    
    with patch.object(ehap, '_fetch_new_token', return_value="mock_token"):
        llm = ChatOpenAI(
            api_key=ehap.get_token(),
            base_url="http://mock-api",
            model="gpt-4"
        )
        
        # Create proper AuthenticationError
        mock_response = MagicMock()
        mock_response.status_code = 401
        auth_error = AuthenticationError(
            "Invalid credentials",
            response=mock_response,
            body={"error": {"message": "Invalid credentials"}}
        )
        
        # Patch at class level
        with patch('langchain_openai.ChatOpenAI.invoke', side_effect=auth_error):
            messages = [{"role": "user", "content": "Test"}]
            
            try:
                # No llm_reinitializer provided
                response, _ = llm_invoke(
                    llm=llm,
                    ehap=ehap,
                    messages=messages,
                    llm_reinitializer=None  # No reinitializer!
                )
                print("❌ Should have raised AuthenticationError!")
                assert False, "Expected AuthenticationError"
            except AuthenticationError:
                print("✅ Correctly raised AuthenticationError when no reinitializer provided")
                print("=" * 80)


if __name__ == "__main__":
    print("\n" + "🧪" * 40)
    print("TOKEN EXPIRY SIMULATION TEST SUITE")
    print("Testing without Redis (local environment)")
    print("🧪" * 40)
    
    try:
        # Test 1: Basic token expiry with retry
        test_token_expiry_with_retry()
        
        # Test 2: Structured output with expiry
        test_structured_output_with_expiry()
        
        # Test 3: Without reinitializer
        test_without_reinitializer()
        
        print("\n\n" + "🎉" * 40)
        print("ALL TESTS PASSED!")
        print("Token expiry handling is working correctly.")
        print("Safe to deploy to server.")
        print("🎉" * 40 + "\n")
        
    except Exception as e:
        print("\n\n" + "❌" * 40)
        print("TEST SUITE FAILED!")
        print(f"Error: {e}")
        print("❌" * 40 + "\n")
        import traceback
        traceback.print_exc()
        exit(1)
