from __future__ import annotations

import unittest

from main import _render_sft_row


def char_tokenize(text: str) -> list[int]:
    return [ord(ch) for ch in text]


class SftRendererTests(unittest.TestCase):
    def test_renders_messages_schema(self) -> None:
        row = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }

        rendered = _render_sft_row(row, char_tokenize, seq_len=1024)

        self.assertIsNotNone(rendered)
        input_ids, labels = rendered
        self.assertEqual(len(input_ids), len(labels))
        supervised = [label for label in labels if label != -100]
        self.assertEqual(supervised, char_tokenize("Hi<|im_end|>\n"))

    def test_still_renders_sharegpt_schema(self) -> None:
        row = {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi"},
            ]
        }

        rendered = _render_sft_row(row, char_tokenize, seq_len=1024)

        self.assertIsNotNone(rendered)
        input_ids, labels = rendered
        self.assertEqual(len(input_ids), len(labels))
        supervised = [label for label in labels if label != -100]
        self.assertEqual(supervised, char_tokenize("Hi<|im_end|>\n"))


if __name__ == "__main__":
    unittest.main()
