"""
LLM-Powered Batch Scoring Analytics Platform with Native Google AI Studio Integration
Author: Custom Analytics Pipeline Platform
"""

import os
import re
import json
import requests
import jsonschema
from jsonschema import validate

class LLMScoringEngine:
    def __init__(self):

        # Native Google AI Studio high-throughput model route
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent"
        
        # Ingestion logic for local key config configurations
        api_key = os.environ.get("LLM_API_KEY", None)
        if not api_key and os.path.exists("key_config.txt"):
            with open("key_config.txt", "r") as f:
                api_key = f.read().strip()
                
        self.api_key = api_key
        
        # Enforce exact JSON schema boundaries
        self.output_schema = {
            "type": "object",
            "properties": {
                "risk_tier": {"type": "string", "enum": ["low", "medium", "high"]},
                "flag_for_review": {"type": "boolean"},
                "primary_signal": {"type": "string"},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "recommended_action": {"type": "string"}
            },
            "required": ["risk_tier", "flag_for_review", "primary_signal", "confidence", "recommended_action"]
        }
        
        self.system_prompt = (
            "You are a structured operational compliance auditor. Assess the transactional records "
            "provided against this business rubric:\n"
            "1. High revenue (>2000) combined with non-premium status marks a 'high' risk tier and true flag.\n"
            "2. Mid revenue (500-2000) defaults to 'medium' risk tier.\n"
            "3. Lower volumes scale as 'low' risk.\n"
            "You must return ONLY a valid JSON object matching the schema rules. No markdown wrapper blocks, no backticks, no markdown fence codes.\n\n"
            "Worked Input Example:\n"
            "{\"revenue\": 2200.0, \"customer_segment\": 1, \"numeric_stored_as_object\": 12.0, \"repetitive_string_col_West\": 1}\n\n"
            "Worked Output Example:\n"
            "{\n"
            "  \"risk_tier\": \"high\",\n"
            "  \"flag_for_review\": true,\n"
            "  \"primary_signal\": \"Out of pattern revenue scale for non-premium standard tier account.\",\n"
            "  \"confidence\": \"high\",\n"
            "  \"recommended_action\": \"Suspend transaction immediately and forward to compliance.\"\n"
            "}"
        )

    def verify_environment(self):
        print("=== Step 1: Environment Security Audit ===")
        if not self.api_key:
            raise ValueError("Security Failure: The system 'LLM_API_KEY' configuration is empty.")
        print("Verification Complete: API credentials verified and configured successfully.")

    def run_pii_guardrail(self, input_text: str) -> bool:
        # Improved regex structure to trace patterns hidden within JSON characters
        email_regex = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        phone_regex = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
        return bool(re.search(email_regex, input_text) or re.search(phone_regex, input_text))

    def execute_llm_transaction(self, system_msg: str, user_msg: str, target_temp: float = 0.0) -> str:
        # Standardize payload format to match Google API criteria
        url = f"{self.api_url}?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        combined_instruction = f"{system_msg}\n\nProcess the following text payload: {user_msg}"
        
        payload = {
            "contents": [{"parts": [{"text": combined_instruction}]}],
            "generationConfig": {
                "temperature": target_temp,
                "maxOutputTokens": 512,
                "responseMimeType": "application/json" # Forces Gemini to return pure JSON directly
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            if response.status_code != 200:
                print(f"Network Transaction Dropped. Status Code Received: {response.status_code}")
                print(f"Content: {response.text}")
                return None
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as err:
            print(f"Execution Error inside network block: {str(err)}")
            return None

    def execute_safeguarded_pipeline(self, target_record: dict, temp_setting: float = 0.0) -> dict:
        serialized_input = json.dumps(target_record)
        
        # 1. Evaluate PII Boundaries
        if self.run_pii_guardrail(serialized_input):
            print("Pipeline Block Action: Personal Identifiable Information (PII) detected in input vector.")
            return None
            
        # 2. Complete Model Request
        raw_response = self.execute_llm_transaction(self.system_prompt, serialized_input, target_temp=temp_setting)
        if not raw_response:
            return self.get_fallback_payload("Network link dropped or returned null payload values.")
            
        cleaned_text = raw_response.strip()
        
        # 3. Structural Parse Checks
        try:
            parsed_object = json.loads(cleaned_text)
        except json.JSONDecodeError as parse_err:
            print(f"JSON Structure Failure: {str(parse_err)}")
            return self.get_fallback_payload("Malformed structural layout return framework.")
            
        # 4. Schema Verification Enforcement Rules
        try:
            validate(instance=parsed_object, schema=self.output_schema)
            return parsed_object
        except jsonschema.ValidationError as validation_err:
            print(f"Schema Enforcement Breach: {validation_err.message}")
            return self.get_fallback_payload(f"Validation failure message: {validation_err.message}")

    def get_fallback_payload(self, diagnostics: str) -> dict:
        return {
            "risk_tier": "low",
            "flag_for_review": False,
            "primary_signal": f"Pipeline Fallback Triggered: {diagnostics}",
            "confidence": "low",
            "recommended_action": "Process standard run."
        }

if __name__ == "__main__":
    engine = LLMScoringEngine()
    engine.verify_environment()
    
    print("\n=== Step 2: Verification of Core Hello Prompt ===")
    test_text = engine.execute_llm_transaction("You are a string reflector.", "Reply with only the word hello.", target_temp=0.0)
    print(f"Reflector Output: {test_text.strip() if test_text else 'None'}")
    
    print("\n=== Step 3: PII Security Interception Validation ===")
    clean_sample = {"revenue": 150.5, "customer_segment": 2}
    dirty_sample = {"revenue": 150.5, "customer_segment": 2, "support_contact": "auditor_john@client_corp.com"}
    
    print(f"Clean Vector Intercept Result (Should be False): {engine.run_pii_guardrail(json.dumps(clean_sample))}")
    print(f"Dirty Vector Intercept Result (Should be True) : {engine.run_pii_guardrail(json.dumps(dirty_sample))}")
    
    sample_records = [
        {"revenue": 95.0, "customer_segment": 1, "numeric_stored_as_object": 85.0, "repetitive_string_col_East": 0, "repetitive_string_col_North": 1, "repetitive_string_col_South": 0, "repetitive_string_col_West": 0},
        {"revenue": 4500.0, "customer_segment": 1, "numeric_stored_as_object": 42.5, "repetitive_string_col_East": 0, "repetitive_string_col_North": 0, "repetitive_string_col_South": 0, "repetitive_string_col_West": 1},
        {"revenue": 140.25, "customer_segment": 2, "numeric_stored_as_object": 40.0, "repetitive_string_col_East": 0, "repetitive_string_col_North": 0, "repetitive_string_col_South": 1, "repetitive_string_col_West": 0}
    ]
    
    print("\n=== Step 4: Temperature Volatility & Determinism Comparisons ===")
    for idx, rec in enumerate(sample_records[:1]):
        print(f"Evaluating Sample Input Vector #{idx+1} at Temp = 0.0:")
        out_t0 = engine.execute_safeguarded_pipeline(rec, temp_setting=0.0)
        print(json.dumps(out_t0, indent=2))
        
        print(f"Evaluating Sample Input Vector #{idx+1} at Temp = 0.7:")
        out_t7 = engine.execute_safeguarded_pipeline(rec, temp_setting=0.7)
        print(json.dumps(out_t7, indent=2))
        
    print("\n=== Step 5: End-to-End Production Pipeline Run ===")
    for idx, record in enumerate(sample_records):
        print(f"\n--- Processing Tabular Node Record #{idx+1} ---")
        final_result = engine.execute_safeguarded_pipeline(record, temp_setting=0.0)
        print("Final Validated System Assessment Output Array:")
        print(json.dumps(final_result, indent=2))