

import unittest
from llm_utils.parsers.cue_parser import CueParser

class TestCueParser(unittest.TestCase):

    def setUp(self):
        self.parser = CueParser()

    def test_happy_case(self):
        input_text = """
        <P1> Scene One
        <P2> Another Scene
        <P10> Final Scene
        """
        expected = [(1, "Scene One"), (2, "Another Scene"), (10, "Final Scene")]
        result = self.parser.parse(input_text)
        self.assertEqual(result, expected)

    def test_sad_case_no_valid_lines(self):
        input_text = """
        This is not a cue line.
        Nor is this.
        """
        with self.assertRaises(ValueError):
            self.parser.parse(input_text)

if __name__ == "__main__":
    unittest.main()