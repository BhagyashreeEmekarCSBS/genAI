"""
Comprehensive Unit Test Suite for AI Multi-Agent Software Development System.
Validates all test cases specified in the requirements:
- Test Case 1: Valid API Key / Connection check
- Test Case 2: Invalid API Key handling
- Test Case 3: Missing / Invalid SRS blocks downstream code generation
- Test Case 4: Invalid Python Code AST detection with line number & suggested fix
- Test Case 5: Valid Python Code AST parsing
- Test Case 6: Empty / Pending Code handling without false syntax errors
- Test Case 7: LangGraph StateGraph validation gates
"""

import os
import sys
import unittest

# Ensure multi_agent_development directory is in search path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from utils.validators import (
    validate_python_code,
    validate_srs,
    validate_source_code,
    validate_code
)
from utils.llm import (
    validate_api_connection,
    list_content_generation_models,
    DEFAULT_GEMINI_MODEL
)
from workflow.graph import (
    create_development_workflow,
    srs_validation_gate,
    code_validation_gate
)
from workflow.state import AgentWorkflowState
from agents.code_review_agent import extract_verdict
from agents.code_generation_agent import extract_code_block


class TestASTValidation(unittest.TestCase):
    """Tests 4 & 5: Deterministic AST Syntax Validation."""

    def test_valid_python_code(self):
        """Test Case 5: Valid Python source code passes AST inspection."""
        code = '''
import os
from datetime import datetime

class StudentManager:
    def __init__(self):
        self.students = {}

    def register_student(self, student_id: str, name: str) -> bool:
        if student_id in self.students:
            return False
        self.students[student_id] = {"name": name, "created": datetime.now()}
        return True

def main():
    manager = StudentManager()
    manager.register_student("S001", "Alice")
'''
        res = validate_python_code(code)
        self.assertTrue(res["is_valid"])
        self.assertTrue(res["has_code"])
        self.assertEqual(res["status"], "PASSED")
        self.assertIn("StudentManager", res["classes"])
        self.assertIn("register_student", res["functions"])
        self.assertIn("main", res["functions"])
        self.assertIn("os", res["imports"])
        self.assertIn("datetime.datetime", res["imports"])

    def test_invalid_python_code_syntax_error(self):
        """Test Case 4: Invalid Python code detects syntax error, line number, and suggested fix."""
        broken_code = '''def calculate_total():
    if True
        return 100'''
        res = validate_python_code(broken_code)
        self.assertFalse(res["is_valid"])
        self.assertTrue(res["has_code"])
        self.assertEqual(res["status"], "FAILED")
        self.assertEqual(res["error_line"], 2)
        self.assertIsNotNone(res["error"])
        self.assertIsNotNone(res["suggested_fix"])
        self.assertIn("colon", res["suggested_fix"].lower())

    def test_empty_code_handling_no_false_error(self):
        """Test Case 6: Empty code displays informative message instead of false syntax error."""
        res = validate_python_code("")
        self.assertFalse(res["is_valid"])
        self.assertFalse(res["has_code"])
        self.assertEqual(res["status"], "Not Executed")
        self.assertIn("Code generation has not completed yet", res["summary"])

        res_none = validate_python_code(None)
        self.assertFalse(res_none["has_code"])
        self.assertIn("Code generation has not completed yet", res_none["summary"])


class TestValidationGatesAndWorkflow(unittest.TestCase):
    """Test Case 3: Missing SRS and Code Validation Gates in LangGraph."""

    def test_srs_validator(self):
        # Valid SRS
        valid_srs = (
            "# Software Requirement Specification (SRS): Library\n\n"
            "## 1. Project Overview\nA system to manage books.\n\n"
            "## 2. Functional Requirements\n- FR-1: Add books\n- FR-2: Issue books\n"
            + ("Detailed specifications and requirements. " * 10)
        )
        is_ok, msg = validate_srs(valid_srs)
        self.assertTrue(is_ok)

        # Invalid SRS (Empty)
        is_ok, msg = validate_srs("")
        self.assertFalse(is_ok)

        # Invalid SRS (API Error message)
        err_srs = "### Error Generating Requirements\n\n400 INVALID_ARGUMENT: API key not valid."
        is_ok, msg = validate_srs(err_srs)
        self.assertFalse(is_ok)

    def test_srs_validation_gate_halts_downstream(self):
        """Verify that failed/missing SRS prevents downstream execution."""
        failed_state: AgentWorkflowState = {
            "srs_status": "Failed",
            "srs_report": None
        }
        decision = srs_validation_gate(failed_state)
        self.assertEqual(decision, "halt_on_srs_failure")

        completed_state: AgentWorkflowState = {
            "srs_status": "Completed",
            "srs_report": "# Software Requirement Specification\n\n## 1. Overview\nValid SRS content.\n" + ("Text " * 40)
        }
        decision = srs_validation_gate(completed_state)
        self.assertEqual(decision, "proceed_to_code_gen")

    def test_code_validation_gate_halts_downstream(self):
        """Verify that failed/missing code prevents review from executing."""
        failed_state: AgentWorkflowState = {
            "code_status": "Failed",
            "generated_code": None
        }
        decision = code_validation_gate(failed_state)
        self.assertEqual(decision, "halt_on_code_failure")

        # Placeholder error code should be rejected
        placeholder_state: AgentWorkflowState = {
            "code_status": "Completed",
            "generated_code": "# Code generation could not proceed: SRS report is missing or invalid."
        }
        decision = code_validation_gate(placeholder_state)
        self.assertEqual(decision, "halt_on_code_failure")

        # Valid code should proceed
        valid_state: AgentWorkflowState = {
            "code_status": "Completed",
            "generated_code": "def hello():\n    print('Hello World')\n\nif __name__ == '__main__':\n    hello()"
        }
        decision = code_validation_gate(valid_state)
        self.assertEqual(decision, "proceed_to_code_review")

    def test_graph_structure(self):
        """Verify compiled StateGraph contains all required nodes."""
        workflow = create_development_workflow()
        self.assertIn("requirement_analysis", workflow.nodes)
        self.assertIn("code_generation", workflow.nodes)
        self.assertIn("code_review", workflow.nodes)


class TestAPIKeyValidation(unittest.TestCase):
    """Test Case 2: Invalid / Missing API Key handling."""

    def test_missing_api_key(self):
        is_valid, msg = validate_api_connection(api_key="your_gemini_api_key_here")
        self.assertFalse(is_valid)
        self.assertIn("GEMINI_API_KEY is missing", msg)

    def test_invalid_api_key(self):
        is_valid, msg = validate_api_connection(api_key="AIzaSyDummyInvalidKeyForTesting12345")
        self.assertFalse(is_valid)
        self.assertTrue("Authentication Error" in msg or "Client Error" in msg or "API key is invalid" in msg)

    def test_dynamic_models_listing(self):
        models = list_content_generation_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        self.assertEqual(models[0], DEFAULT_GEMINI_MODEL)
        self.assertIn("gemini-2.5-flash-lite", models)


class TestParsers(unittest.TestCase):
    def test_verdict_extraction(self):
        self.assertEqual(extract_verdict("Final Verdict: Approved"), "Approved")
        self.assertEqual(extract_verdict("Final Verdict: Approved with Suggestions"), "Approved with Suggestions")
        self.assertEqual(extract_verdict("Final Verdict: Needs Improvement"), "Needs Improvement")

    def test_code_block_extraction(self):
        markdown_text = """
```python
def add(a, b):
    return a + b
```
"""
        code, lang = extract_code_block(markdown_text, "python")
        self.assertEqual(lang, "python")
        self.assertIn("def add(a, b):", code)


if __name__ == "__main__":
    unittest.main()
