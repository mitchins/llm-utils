import re
from typing import List, Tuple

class CueParser:
    CUE_PATTERN = re.compile(r"<P(\d+)>\s+(.*)")

    def parse(self, cue_text: str) -> List[Tuple[int, str]]:
        """
        Parses cue lines in the format <P{number}> Scene Title and returns a list of tuples.

        Args:
            cue_text (str): Multiline string containing the cues.

        Returns:
            List[Tuple[int, str]]: List of (paragraph number, scene title) tuples.
        """
        cues = []
        for line in cue_text.strip().splitlines():
            match = self.CUE_PATTERN.match(line.strip())
            if match:
                para_index = int(match.group(1))
                scene_name = match.group(2).strip()
                cues.append((para_index, scene_name))
        if not cues:
            raise ValueError("No valid cue lines found in the response. Expected format: <P{number}> Scene Title")
        return cues
