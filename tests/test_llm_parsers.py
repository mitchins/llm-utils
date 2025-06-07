import unittest
from llm_utils.llm_parsers import process_llm_response, LLMResponse, extract_json_array
from unittest.mock import MagicMock


# Helper for invalid LLM response test cases
def run_invalid_llm_test_case(test_input: str):
    console = MagicMock()
    result = process_llm_response(test_input, console)
    assert result == LLMResponse.FAILED
    console.print.assert_called_with(f"[bold yellow]Rejected:[/bold yellow] Invalid response format: {test_input.strip()}", style="bold yellow")

def run_llm_test_case(test_input: str, expected: LLMResponse):
    console = MagicMock()
    result = process_llm_response(test_input, console)
    assert result == expected, f"Got {result}, expected {expected} for input: {repr(test_input)}"
    console.print.assert_not_called()

class TestProcessLLMResponse(unittest.TestCase):
    def test_invalid_answer_prefix_with_no(self):
        response = """**Answer:**  
No.  

The provided text contains multiple `PARASEP_CONTINUE` segments, indicating that each paragraph is a continuation of the 
previous beat. The final `[SEP]` is likely a BERT-specific separator token (not a user-defined `PARASEP_CHANGE`) and does not 
indicate a scene shift. Therefore, there is **no** `PARASEP_CHANGE` in this text."""  
        result = process_llm_response(response, self.console)
        self.assertEqual(result, LLMResponse.NEGATIVE)
        self.console.print.assert_not_called()

    def setUp(self):
        self.console = MagicMock()

    def test_affirmative_brackets_only(self):
        run_llm_test_case("[Y]", LLMResponse.AFFIRMATIVE)

    def test_affirmative_double_asterisks(self):
        run_llm_test_case("**[Y]**", LLMResponse.AFFIRMATIVE)

    def test_affirmative_double_asterisks_no_brackets(self):
        run_llm_test_case("**Y**", LLMResponse.AFFIRMATIVE)

    def test_affirmative_single_quotes(self):
        run_llm_test_case("'Y'", LLMResponse.AFFIRMATIVE)

    def test_affirmative_double_brackets(self):
        run_llm_test_case("[[Y]]", LLMResponse.AFFIRMATIVE)

    def test_negative_plain(self):
        run_llm_test_case("N", LLMResponse.NEGATIVE)

    def test_negative_with_spaces(self):
        run_llm_test_case("  [N]  ", LLMResponse.NEGATIVE)

    def test_negative_asterisks_and_spaces(self):
        run_llm_test_case("  **[N]**  ", LLMResponse.NEGATIVE)

    def test_affirmative_newline(self):
        run_llm_test_case("[Y]\n", LLMResponse.AFFIRMATIVE)

    def test_affirmative_asterisks_newline(self):
        run_llm_test_case("**[Y]**\n", LLMResponse.AFFIRMATIVE)

    def test_affirmative_quotes_newline(self):
        run_llm_test_case("'Y'\n", LLMResponse.AFFIRMATIVE)

    def test_affirmative_double_brackets_newline(self):
        run_llm_test_case("[[Y]]\n", LLMResponse.AFFIRMATIVE)

    def test_negative_newline(self):
        run_llm_test_case("N\n", LLMResponse.NEGATIVE)

    def test_negative_spaces_newline(self):
        run_llm_test_case("  [N]  \n", LLMResponse.NEGATIVE)

    def test_negative_asterisks_spaces_newline(self):
        run_llm_test_case("  **[N]**  \n", LLMResponse.NEGATIVE)

    def test_affirmative_quotes_inside_brackets(self):
        run_llm_test_case("[ ' Y ' ]", LLMResponse.AFFIRMATIVE)

    def test_affirmative_asterisk_bracket_combo(self):
        run_llm_test_case("* [Y]*", LLMResponse.AFFIRMATIVE)

    def test_affirmative_trailing_space(self):
        run_llm_test_case("[Y ]", LLMResponse.AFFIRMATIVE)

    def test_negative_trailing_space(self):
        run_llm_test_case("[N ]", LLMResponse.NEGATIVE)

    def test_affirmative_spaces_no_brackets(self):
        run_llm_test_case(" Y ", LLMResponse.AFFIRMATIVE)

    def test_negative_leading_space(self):
        run_llm_test_case(" N", LLMResponse.NEGATIVE)

    def test_affirmative_brackets_with_spaces(self):
        run_llm_test_case("[ Y ]", LLMResponse.AFFIRMATIVE)

    def test_affirmative_single_quotes_inside_brackets(self):
        run_llm_test_case("['Y']", LLMResponse.AFFIRMATIVE)

    def test_affirmative_nested_quotes_brackets(self):
        run_llm_test_case("[[ 'Y' ]]", LLMResponse.AFFIRMATIVE)

    def test_negative_quotes_with_brackets(self):
        run_llm_test_case("' [N] '", LLMResponse.NEGATIVE)

    def test_negative_quotes_inside_brackets(self):
        run_llm_test_case("[ ' N ' ]", LLMResponse.NEGATIVE)

    def test_affirmative_with_reasoning_text(self):
        run_llm_test_case("**[Y]**\nABC", LLMResponse.AFFIRMATIVE)

    def test_negative_answer_prefix_1(self):
        run_llm_test_case("**Answer:**\n[N]\n\nThe scene continues within the same beat. The text describes Kiki cutting and eating an apple, which is a natural progression of the previous actions (e.g., Levi slicing strawberries, Howard managing trash). There’s no abrupt shift in setting, perspective, or narrative focus—only a continuation of the domestic, quiet moment. The [SEP] likely marks the end of the training segment but does not indicate a scene change.", LLMResponse.NEGATIVE)

    def test_negative_answer_prefix_2(self):
        run_llm_test_case("**Answer:**\n[N]\n\nThe text provided does **not** contain a `PARASEP_CHANGE`. The narrative continues within the same beat, with each `[PARASEP_CONTINUE]` indicating progression in the current scene. The shift occurs only when a new beat or scene begins (marked by `PARASEP_CHANGE`). In this case, the story flows seamlessly from Kiki's approach to the woman, their interaction, and the resolution of that moment without introducing a new scene.", LLMResponse.NEGATIVE)

    def test_affirmative_with_extra_text(self):
        run_llm_test_case("[Y] some extra text", LLMResponse.AFFIRMATIVE)

    def test_negative_double_asterisks_no_brackets(self):
        run_llm_test_case("**N**", LLMResponse.NEGATIVE)

    def test_invalid_response_with_unknown_bracket(self):
        run_invalid_llm_test_case("[X]")

    def test_invalid_response_with_text_after_y(self):
        run_llm_test_case("Y something else", LLMResponse.AFFIRMATIVE)

    def test_invalid_response_with_bracketed_unknown(self):
        run_invalid_llm_test_case("[UNKNOWN]")

    def test_invalid_response_with_answer_prefix_and_no(self):
        run_llm_test_case("**Answer:** No (N)", LLMResponse.NEGATIVE)

    def test_complex_valid_responses(self):
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
        result = process_llm_response(response, self.console)

        # Assert the function returns the expected status
        self.assertEqual(result, LLMResponse.AFFIRMATIVE)

        # Ensure no console print happens for valid responses
        self.console.print.assert_not_called()

    def test_should_scene_contain_change_prefix(self):
        response = """Some original prompt [Y/N]: **N**

Some long winded justification."""
        result = process_llm_response(response, self.console)
        self.assertEqual(result, LLMResponse.NEGATIVE)
        self.console.print.assert_not_called()

    def test_should_scene_contain_change_prefix_with_explanation(self):
        response = """Some original prompt [Y/N]: **N**

**Explanation:**  
Some long winded justification."""
        result = process_llm_response(response, self.console)
        self.assertEqual(result, LLMResponse.NEGATIVE)
        self.console.print.assert_not_called()

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