# What Happens When Fresh Token Also Fails?

## The Question

**"What if the new LLM client with fresh token also fails? How are you handling that scenario?"**

This is a critical edge case that needs proper handling.

---

## Current Implementation

### Two-Level Retry Strategy

The code has **TWO layers of retry**:

1. **Manual Immediate Retry** (inside the exception handler)
2. **Tenacity Automatic Retry** (decorator-based)

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. First Attempt with Original LLM                          │
├─────────────────────────────────────────────────────────────┤
│ llm.invoke(messages)                                        │
│ ↓                                                           │
│ Result?                                                     │
└─────────────────────────────────────────────────────────────┘
                    ↓                    ↓
              ✅ SUCCESS            ❌ 401 ERROR
                    ↓                    ↓
          Return result      ┌──────────────────────────┐
                            │ 2. Manual Retry Handler   │
                            ├──────────────────────────┤
                            │ • Force token refresh    │
                            │ • Create NEW LLM         │
                            │ • Retry immediately      │
                            └──────────────────────────┘
                                       ↓
                                  Result?
                    ↓                                    ↓
              ✅ SUCCESS                          ❌ STILL 401
                    ↓                                    ↓
          Return result                    ┌──────────────────────────┐
                                          │ 3. Re-raise Exception     │
                                          ├──────────────────────────┤
                                          │ raise retry_error        │
                                          └──────────────────────────┘
                                                       ↓
                                          ┌──────────────────────────┐
                                          │ 4. Tenacity Catches It   │
                                          ├──────────────────────────┤
                                          │ Retry decorator kicks in │
                                          │ max_attempts=2           │
                                          └──────────────────────────┘
                                                       ↓
                                          ┌──────────────────────────┐
                                          │ 5. Tenacity Retry        │
                                          ├──────────────────────────┤
                                          │ Calls entire function    │
                                          │ again from the start     │
                                          └──────────────────────────┘
                                                       ↓
                                                  Result?
                                    ↓                                ↓
                              ✅ SUCCESS                      ❌ STILL FAILS
                                    ↓                                ↓
                          Return result              ┌──────────────────────┐
                                                    │ 6. Final Failure     │
                                                    ├──────────────────────┤
                                                    │ Tenacity re-raises   │
                                                    │ Exception propagates │
                                                    │ to caller            │
                                                    └──────────────────────┘
```

---

## Code Walkthrough

### Level 1: Manual Immediate Retry

**File:** `ehap_retry.py` Lines 146-172

```python
except AuthenticationError:
    # Force token refresh for retry
    logger.warning("AuthenticationError caught. Forcing token refresh for retry.")
    ehap.force_token_refresh()
    
    # If reinitializer provided, recreate LLM with fresh token and retry immediately
    if llm_reinitializer:
        llm = llm_reinitializer()  # ← Create NEW LLM with fresh token
        logger.debug("LLM reinitialized with fresh token for retry")
        
        # Retry with new LLM client (don't re-raise, do actual retry here)
        try:
            result = llm.invoke(messages, **invoke_kwargs)  # ← RETRY #1
            
            # Clear bypass flag after successful retry
            from deep_research_utils.cache_utils import clear_bypass_flag
            clear_bypass_flag()
            
            logger.info("Retry with fresh token succeeded")
            return result, llm  # ← SUCCESS - return immediately
            
        except AuthenticationError as retry_error:
            # ⚠️ THIS IS WHERE WE HANDLE FRESH TOKEN FAILURE
            # If still failing after refresh, re-raise to trigger tenacity retry
            logger.error("AuthenticationError persists after token refresh")
            raise retry_error  # ← Re-raise for Tenacity to handle
    else:
        # No reinitializer, re-raise to trigger tenacity retry with same LLM
        raise
```

**What happens here:**
1. ✅ First 401 → Create new LLM → Retry immediately
2. ✅ If retry succeeds → Return success
3. ⚠️ **If retry ALSO fails → Re-raise exception**

---

### Level 2: Tenacity Automatic Retry

**File:** `ehap_retry.py` Lines 48-58

```python
def _create_retry_decorator(max_attempts: int = 2):
    """Create a tenacity retry decorator for EHAP token errors."""
    return retry(
        retry=retry_if_exception_type(AuthenticationError),  # ← Only retry on 401
        stop=stop_after_attempt(max_attempts),  # ← Max 2 attempts total
        before_sleep=_before_retry_callback,  # ← Log before retry
        reraise=True  # ← Re-raise if all attempts fail
    )

@_create_retry_decorator(max_attempts=2)
def llm_invoke(...):
    # Function body
```

**What happens here:**
1. Decorator wraps the entire function
2. If `AuthenticationError` is raised (from Level 1)
3. Tenacity catches it
4. Calls the **entire function again** (fresh start)
5. Max 2 total attempts
6. If both fail → Re-raise to caller

---

## Detailed Scenario: Fresh Token Also Fails

### Attempt 1: Original Token

```
Time: 11:00 AM
Token: token_A (from 07:00 AM, now expired)

[Attempt 1.1] llm.invoke() with token_A
❌ 401 AuthenticationError

[Manual Retry Handler]
→ ehap.force_token_refresh()
→ Fetch fresh token_B from EHAP
→ Create new LLM with token_B

[Attempt 1.2] llm.invoke() with token_B
❌ 401 AuthenticationError (STILL FAILS!)

→ Log: "AuthenticationError persists after token refresh"
→ raise retry_error
```

### Attempt 2: Tenacity Retry

```
[Tenacity Catches Exception]
→ Log: "Retry attempt 2 after AuthenticationError. Refreshing EHAP token..."
→ Calls llm_invoke() again from the start

[Attempt 2.1] llm.invoke() with token_B (or fetch token_C)
❌ 401 AuthenticationError

[Manual Retry Handler]
→ ehap.force_token_refresh()
→ Fetch fresh token_C from EHAP
→ Create new LLM with token_C

[Attempt 2.2] llm.invoke() with token_C
❌ 401 AuthenticationError (STILL FAILS!)

→ Log: "AuthenticationError persists after token refresh"
→ raise retry_error
```

### Final Failure

```
[Tenacity Max Attempts Reached]
→ max_attempts=2 exhausted
→ reraise=True
→ AuthenticationError propagates to caller

[Agent Error Handler]
→ Catches exception
→ Returns failed status to API
→ User sees error response
```

---

## Total Retry Count

| Attempt | Description | Token Used |
|---------|-------------|------------|
| 1.1 | First try with original LLM | token_A (old) |
| 1.2 | Manual retry with fresh LLM | token_B (fresh) |
| 2.1 | Tenacity retry, original LLM | token_B or token_C |
| 2.2 | Manual retry with fresh LLM | token_C (fresh) |

**Maximum: 4 LLM invocations** (2 manual + 2 tenacity)

---

## Why Would Fresh Token Fail?

### Possible Reasons:

1. **EHAP Server Down**
   - EHAP OAuth endpoint unavailable
   - Returns 500/503 errors
   - Network timeout

2. **Invalid Credentials**
   - `EHAP_CLIENT_ID` or `EHAP_CLIENT_SECRET` changed
   - Credentials revoked
   - Wrong environment configuration

3. **OpenAI Endpoint Issue**
   - `OPENAI_BASE_URL` misconfigured
   - Proxy/firewall blocking requests
   - Corporate network issues

4. **Token Format Mismatch**
   - EHAP returns token in unexpected format
   - Token encoding issues
   - JWT parsing errors

5. **Rate Limiting**
   - Too many token requests
   - EHAP throttling
   - OpenAI rate limits

6. **Concurrent Token Rotation**
   - Multiple processes requesting tokens
   - Race condition in token storage
   - Cache coherency issues

---

## Error Handling Flow

### In Agent (AgentBase)

**File:** `base_agent.py` Lines 560-600

```python
def __call__(self, **kwargs: Any) -> Dict[str, Any]:
    """Execute the agent."""
    try:
        # Prepare state
        state = self.prepare_state(**kwargs)
        
        # Execute graph
        result = self.app.invoke(state)
        
        # Extract result
        return self.extract_result(result)
        
    except Exception as exc:
        # ⚠️ THIS CATCHES THE FINAL FAILURE
        self.logger.error(f"Agent execution failed: {exc}", exc_info=True)
        return self.handle_execution_error(exc, **kwargs)
```

### In API Response

**File:** `pattern_agent.py` Lines 2015-2055

```python
def handle_execution_error(self, exc: Exception, **kwargs: Any) -> PatternAgentOutput:
    """Handle execution errors and return failed response."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "job_id": str(kwargs.get("job_id") or uuid.uuid4().hex),
        "conversation_id": kwargs.get("conversation_id"),
        "agent": PATTERN_AGENT_NAME,
        "status": "failed",  # ← Status set to failed
        "output": {
            "business_patterns": [],
            "executive_summary": {...},
            "quality_checks": {...},
            # ... empty output ...
        },
        "visual_component": {},
        "explanation": {"error": str(exc)},  # ← Error message included
        "validation": {
            "is_valid": False,
            "checks": [],
            "warnings": [],
            "errors": [str(exc)],  # ← Error details
        },
        "tokens": {"input": 0, "output": 0, "breakdown": {}},
        "execution": {
            "start_time": now,
            "end_time": now,
            "duration_ms": 0,
            "version": PATTERN_AGENT_VERSION,
        },
    }
```

---

## Logs You'll See

### Successful Retry (Fresh Token Works)

```
2026-06-16 11:00:00 - WARNING - AuthenticationError caught. Forcing token refresh for retry.
2026-06-16 11:00:00 - INFO - Clearing token cache and setting Redis bypass flag...
2026-06-16 11:00:00 - INFO - Token not in cache. Fetching new token.
2026-06-16 11:00:00 - INFO - Access token generated successfully
2026-06-16 11:00:00 - DEBUG - LLM reinitialized with fresh token for retry
2026-06-16 11:00:01 - INFO - Retry with fresh token succeeded ✅
```

### Failed Retry (Fresh Token Also Fails)

```
2026-06-16 11:00:00 - WARNING - AuthenticationError caught. Forcing token refresh for retry.
2026-06-16 11:00:00 - INFO - Clearing token cache and setting Redis bypass flag...
2026-06-16 11:00:00 - INFO - Token not in cache. Fetching new token.
2026-06-16 11:00:00 - INFO - Access token generated successfully
2026-06-16 11:00:00 - DEBUG - LLM reinitialized with fresh token for retry
2026-06-16 11:00:01 - ERROR - AuthenticationError persists after token refresh ❌
2026-06-16 11:00:01 - WARNING - Retry attempt 2 after AuthenticationError. Refreshing EHAP token...
2026-06-16 11:00:01 - WARNING - AuthenticationError caught. Forcing token refresh for retry.
2026-06-16 11:00:01 - INFO - Clearing token cache and setting Redis bypass flag...
2026-06-16 11:00:01 - INFO - Token not in cache. Fetching new token.
2026-06-16 11:00:01 - INFO - Access token generated successfully
2026-06-16 11:00:01 - DEBUG - LLM reinitialized with fresh token for retry
2026-06-16 11:00:02 - ERROR - AuthenticationError persists after token refresh ❌
2026-06-16 11:00:02 - ERROR - Agent execution failed: Invalid authentication credentials
```

---

## API Response on Final Failure

```json
{
  "job_id": "abc123...",
  "conversation_id": "test_123",
  "agent": "pattern_agent",
  "status": "failed",
  "output": {
    "business_patterns": [],
    "executive_summary": {
      "headline": "",
      "primary_business_message": "",
      "recommended_focus_order": []
    },
    "quality_checks": {
      "patterns_returned": 0,
      "groups_consumed": 0,
      "ungrouped_group_ids": [],
      "notes": []
    },
    "cards": [],
    "groups": [],
    "stats": {},
    "semantic_summary": {}
  },
  "visual_component": {},
  "explanation": {
    "error": "Invalid authentication credentials"
  },
  "validation": {
    "is_valid": false,
    "checks": [],
    "warnings": [],
    "errors": ["Invalid authentication credentials"]
  },
  "tokens": {
    "input": 0,
    "output": 0,
    "breakdown": {}
  },
  "execution": {
    "start_time": "2026-06-16T11:00:02Z",
    "end_time": "2026-06-16T11:00:02Z",
    "duration_ms": 0,
    "version": "1.0.0"
  }
}
```

---

## Monitoring & Alerting

### Metrics to Track

1. **401 Error Rate**
   - Normal: 0-1% (token rotation events)
   - Warning: >5% (potential issue)
   - Critical: >10% (major problem)

2. **Retry Success Rate**
   - Normal: >95% (most retries succeed)
   - Warning: <90% (investigate)
   - Critical: <80% (urgent)

3. **Token Fetch Failures**
   - Normal: 0 per hour
   - Warning: >1 per hour
   - Critical: >5 per hour

4. **Final Failure Rate**
   - Normal: <0.1% (very rare)
   - Warning: >0.5%
   - Critical: >1%

### Alert Conditions

```python
# Example monitoring logic
if retry_failure_rate > 0.5:
    alert("Token retry failures increasing", severity="warning")

if final_failure_rate > 0.1:
    alert("Authentication failures not recovering", severity="critical")

if ehap_fetch_errors > 5:
    alert("EHAP token fetch failing repeatedly", severity="critical")
```

---

## Mitigation Strategies

### 1. Circuit Breaker Pattern

```python
class TokenCircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
            else:
                raise CircuitBreakerOpen("Too many token failures")
        
        try:
            result = func()
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except AuthenticationError:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise
```

### 2. Fallback Token Source

```python
def get_token_with_fallback(ehap_primary, ehap_backup):
    """Try primary EHAP, fallback to backup if needed."""
    try:
        return ehap_primary.get_token()
    except Exception as e:
        logger.warning(f"Primary EHAP failed: {e}, trying backup...")
        return ehap_backup.get_token()
```

### 3. Token Pre-warming

```python
def prewarm_token_cache():
    """Fetch and cache token before it's needed."""
    try:
        ehap = EHAPBase()
        token = ehap.get_token()
        logger.info("Token cache pre-warmed successfully")
    except Exception as e:
        logger.error(f"Token pre-warming failed: {e}")
```

---

## Summary

### What Happens When Fresh Token Fails?

1. **First Failure:** Old token → 401
2. **Manual Retry:** Fresh token → 401 (still fails)
3. **Re-raise:** Exception bubbles up
4. **Tenacity Retry:** Entire function retries
5. **Second Manual Retry:** Another fresh token → 401
6. **Final Failure:** Exception propagates to agent
7. **Error Response:** Agent returns failed status with error details

### Maximum Attempts

- **4 total LLM invocations**
- **3 token fetches** (initial + 2 retries)
- **2 tenacity attempts**
- **2 manual retries**

### Graceful Degradation

✅ **User gets clear error message**
✅ **No infinite loops**
✅ **Proper logging for debugging**
✅ **Failed status in API response**
✅ **Error details preserved**

### When This Happens

This scenario is **very rare** in production and indicates:
- EHAP infrastructure issue
- Configuration problem
- Network/connectivity issue
- Credentials invalid

**Not a bug in your code** - it's a legitimate system failure that should be investigated by ops/infrastructure team.
