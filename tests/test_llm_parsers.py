import unittest
from llm_utils.llm_parsers import process_llm_response, LLMResponse, extract_json_array
from unittest.mock import MagicMock

class TestProcessLLMResponse(unittest.TestCase):

    def test_valid_response(self):
        # Create a mock console object
        console = MagicMock()

        # Test cases with various valid inputs
        test_cases = [
            ("[Y]", LLMResponse.AFFIRMATIVE),  # Single bracketed Y
            ("**[Y]**", LLMResponse.AFFIRMATIVE),  # Double asterisks and square brace around Y
            ("**Y**", LLMResponse.AFFIRMATIVE),  # Double asterisks around Y
            ("'Y'", LLMResponse.AFFIRMATIVE),  # Single quotes around Y
            ("[[Y]]", LLMResponse.AFFIRMATIVE),  # Double brackets around Y
            ("N", LLMResponse.NEGATIVE),  # Plain N
            ("  [N]  ", LLMResponse.NEGATIVE),  # N with spaces around it
            ("  **[N]**  ", LLMResponse.NEGATIVE),  # N with extra symbols and spaces
            ("[Y]\n", LLMResponse.AFFIRMATIVE),  # Y with newline at the end
            ("**[Y]**\n", LLMResponse.AFFIRMATIVE),  # Y with extra symbols and newline
            ("'Y'\n", LLMResponse.AFFIRMATIVE),  # Y with single quotes and newline
            ("[[Y]]\n", LLMResponse.AFFIRMATIVE),  # Y with double brackets and newline
            ("N\n", LLMResponse.NEGATIVE),  # N with newline at the end
            ("  [N]  \n", LLMResponse.NEGATIVE),  # N with spaces and newline
            ("  **[N]**  \n", LLMResponse.NEGATIVE),  # N with extra symbols and newline
            ("[ ' Y ' ]", LLMResponse.AFFIRMATIVE),  # Y surrounded by quotes inside brackets
            ("* [Y]*", LLMResponse.AFFIRMATIVE),  # Y with extra symbols inside brackets
            ("[Y ]", LLMResponse.AFFIRMATIVE),  # Extra space after Y inside brackets
            ("[N ]", LLMResponse.NEGATIVE),  # Extra space after N inside brackets
            (" Y ", LLMResponse.AFFIRMATIVE),  # Y with spaces but no brackets or quotes
            (" N", LLMResponse.NEGATIVE),  # N with leading space
            ("[ Y ]", LLMResponse.AFFIRMATIVE),  # Y inside brackets with spaces
            ("['Y']", LLMResponse.AFFIRMATIVE),  # Y in single quotes inside brackets
            ("[[ 'Y' ]]", LLMResponse.AFFIRMATIVE),  # Y with quotes and brackets
            ("' [N] '", LLMResponse.NEGATIVE),  # N surrounded by extra spaces and quotes
            ("[ ' N ' ]", LLMResponse.NEGATIVE),  # N surrounded by quotes inside brackets
            ("**[Y]**\nABC", LLMResponse.AFFIRMATIVE),  # Y with double asterisks, brackets, and extra text
            ("**Answer:**\n[N]\n\nThe scene continues within the same beat. The text describes Kiki cutting and eating an apple, which is a natural progression of the previous actions (e.g., Levi slicing strawberries, Howard managing trash). There’s no abrupt shift in setting, perspective, or narrative focus—only a continuation of the domestic, quiet moment. The [SEP] likely marks the end of the training segment but does not indicate a scene change.", LLMResponse.NEGATIVE),
            ("**Answer:**\n[N]\n\nThe text provided does **not** contain a `PARASEP_CHANGE`. The narrative continues within the same beat, with each `[PARASEP_CONTINUE]` indicating progression in the current scene. The shift occurs only when a new beat or scene begins (marked by `PARASEP_CHANGE`). In this case, the story flows seamlessly from Kiki's approach to the woman, their interaction, and the resolution of that moment without introducing a new scene.", LLMResponse.NEGATIVE),
        ]

        for response, expected_status in test_cases:
            # Call the function with each test case
            result = process_llm_response(response, console)

            if result != expected_status:
                print(response, expected_status, result)
            # Assert the function returns the expected status
            self.assertEqual(result, expected_status)

            # Ensure no console print happens for valid responses
            console.print.assert_not_called()

    def test_invalid_response(self):
        # Create a mock console object
        console = MagicMock()

        # Test invalid responses
        invalid_responses = [
            "[Y] some extra text",
            "[X]",
            "Y something else",
            "[UNKNOWN]"
        ]

        for response in invalid_responses:
            # Call the function with each invalid test case
            result = process_llm_response(response, console)

            # Assert the function returns FAILED status
            self.assertEqual(result, LLMResponse.FAILED)

            # Check that console.print was called for invalid input
            console.print.assert_called_with(f"[bold yellow]Rejected:[/bold yellow] Invalid response format: {response.strip()}", style="bold yellow")

    def test_complex_valid_responses(self):
        # Create a mock console object
        console = MagicMock()

        response = """
[Y]

**Reasoning:**

The passage clearly demonstrates a shift in mood and focus. The first part is filled with Sassinak's euphoria and self-admiration regarding her graduation. There's a strong sense of personal triumph and a focus on her
appearance and achievements.

The second part abruptly shifts to a darker, more somber tone. The focus moves from Sassinak's personal feelings to the suffering of others ("another colony plundered," "girls like her," "people, real people, murdered and
enslaved").  The mention of Abe and her thoughts about what he "saved" also hint at a deeper, potentially troubling backstory that contrasts with her current celebratory state. This change in subject matter and emotional weight
constitutes a scene shift.



Therefore, a `PARASEP_CHANGE` is appropriate to mark this transition.
"""
        # Call the function with the complex test case
        result = process_llm_response(response, console)

        # Assert the function returns the expected status
        self.assertEqual(result, LLMResponse.AFFIRMATIVE)

        # Ensure no console print happens for valid responses
        console.print.assert_not_called()

class TestJsonExtractionUtility(unittest.TestCase):
    def test_plain_json_only(self):
        text = '[{"name": "Alice", "aliases": []}, {"name": "Bob", "aliases": []}]'
        result = extract_json_array(text)
        self.assertEqual(result, text)

    def test_fenced_json_block(self):
        text = """```json
          [
              {"name": "Alice", "aliases": []},
              {"name": "Bob", "aliases": []}
          ]
          ```"""
        expected = '[\n              {"name": "Alice", "aliases": []},\n              {"name": "Bob", "aliases": []}\n          ]'
        result = extract_json_array(text)
        self.assertEqual(result.strip(), expected.strip())

    def test_json_with_prefix_text(self):
        text = "Here you go:\n[\n  {\"name\": \"Alice\", \"aliases\": []},\n  {\"name\": \"Bob\", \"aliases\": []}\n]"
        expected = '[\n  {"name": "Alice", "aliases": []},\n  {"name": "Bob", "aliases": []}\n]'
        result = extract_json_array(text)
        self.assertEqual(result.strip(), expected.strip())

if __name__ == '__main__':
    unittest.main()