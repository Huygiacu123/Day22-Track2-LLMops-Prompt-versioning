"""
Step 4 — Guardrails AI Validators with Google Gemini
======================================================
TASK:
  1. Build a PIIDetector validator that detects & redacts emails, phone, SSN, credit card
  2. Build a JSONFormatter validator that auto-repairs malformed JSON
  3. Wrap each with a Guard and test with sample inputs
  4. Run a full demo with 6 PII cases and 5 JSON cases

DELIVERABLE: All test cases pass (PII redacted, JSON repaired)
"""

# Fix Unicode encoding on Windows FIRST
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import re
import json

# ── 1. Imports ───────────────────────────────────────────────────────────────
from guardrails import Guard
from guardrails.validators import (
    Validator,
    register_validator,
    PassResult,
    FailResult,
)
from guardrails.validator_base import OnFailAction


# ── 2. PII Detector Validator ─────────────────────────────────────────────────
@register_validator(name="pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Detects and redacts Personally Identifiable Information (PII).
    
    Patterns detected:
      - EMAIL: xxx@xxx.xxx
      - PHONE: (123) 456-7890 or 123-456-7890
      - SSN:   123-45-6789
      - CREDIT CARD: 1234 5678 9012 3456 (or dashes)
    """
    
    PII_PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }
    
    def validate(self, value: str, metadata: dict):
        """
        Check value for PII; if found, redact and return PassResult.
        """
        redacted_text = value
        found_pii = []
        
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, value)
            for match in matches:
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))
        
        if found_pii:
            print(f"  [REDACTED] {len(found_pii)} PII items: {[p[0] for p in found_pii]}")
            return PassResult(value_override=redacted_text)
        
        return PassResult(value_override=value)


# ── 3. JSON Formatter Validator ───────────────────────────────────────────────
@register_validator(name="json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validates and auto-repairs malformed JSON strings.
    
    Common repairs:
      - Strip markdown code fences (``` or ```json)
      - Replace single quotes with double quotes
      - Remove trailing commas before } or ]
      - Re-serialize with json.dumps for consistent formatting
    """
    
    @staticmethod
    def _repair(text: str) -> str:
        """
        Attempt to repair a JSON string.
        """
        text = text.strip()
        
        # Remove markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()
        
        # Single quotes -> double quotes
        text = text.replace("'", '"')
        
        # Remove trailing commas
        text = re.sub(r',\s*([}\]])', r'\1', text)
        
        return text
    
    def validate(self, value: str, metadata: dict):
        """
        Try to parse value as JSON.
        If it fails, try _repair() then parse again.
        """
        # Try parsing as-is
        try:
            parsed = json.loads(value)
            repaired = json.dumps(parsed, indent=2)
            return PassResult(value_override=repaired)
        except json.JSONDecodeError:
            pass
        
        # Try repair
        try:
            repaired_text = self._repair(value)
            parsed = json.loads(repaired_text)
            repaired = json.dumps(parsed, indent=2)
            print(f"  [REPAIRED] JSON fixed successfully")
            return PassResult(value_override=repaired)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON after repair attempt: {str(e)[:50]}"
            print(f"  [FAILED] {error_msg}")
            return FailResult(error_message=error_msg)


# ── 4. PII Guard demo ────────────────────────────────────────────────────────
def demo_pii_guard():
    """
    Create a Guard with PIIDetector and test 6 sample texts.
    """
    print("\n" + "=" * 70)
    print("  PII Detection Demo")
    print("=" * 70)
    
    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
    
    test_cases = [
        ("Email", "Contact John at john.doe@example.com for details."),
        ("Phone", "Call our support line at (555) 867-5309."),
        ("SSN", "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII", "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean", "No sensitive information in this text."),
    ]
    
    for label, text in test_cases:
        result = guard.validate(text)
        status = "PASS" if result.validation_passed else "FAIL"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


# ── 5. JSON Guard demo ────────────────────────────────────────────────────────
def demo_json_guard():
    """
    Create a Guard with JSONFormatter and test 5 sample strings.
    """
    print("\n" + "=" * 70)
    print("  JSON Formatting Demo")
    print("=" * 70)
    
    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))
    
    test_cases = [
        ("Valid JSON", '{"name": "Alice", "age": 30}'),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```'),
        ("Single quotes", "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma", '{"key": "value",}'),
        ("Truly invalid", "This is not JSON at all: ??? {]"),
    ]
    
    for label, text in test_cases:
        result = guard.validate(text)
        status = "PASS" if result.validation_passed else "FAIL"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:60]}")
        print(f"  Output: {str(result.validated_output)[:80]}")


# ── 6. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 70)
    
    demo_pii_guard()
    demo_json_guard()
    
    print("\n" + "=" * 70)
    print("[OK] Step 4 complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
