import unittest
from unittest.mock import patch
from src.translator import translate_content

class TestTranslateContent(unittest.TestCase):

    def test_chinese_message(self):
        is_english, translated_content = translate_content("这是一条中文消息")
        self.assertFalse(is_english, "Expected non-English response for Chinese message")
        self.assertEqual(translated_content, "This is a Chinese message", "Expected Chinese message translation")

    def test_english_message(self):
        is_english, translated_content = translate_content("This is an English message")
        self.assertTrue(is_english, "Expected English detection for English message")
        self.assertEqual(translated_content, "This is an English message", "Expected message to remain unchanged")

    def test_gibberish_message(self):
        is_english, translated_content = translate_content("*q?")
        self.assertTrue(is_english, "Expected English detection for unknown input")
        self.assertEqual(translated_content, "*q?", "Expected gibberish message to remain unchanged")

    @patch('src.translator.translate_content')
    def test_mock_unexpected_response(self, mock_translate):
        is_english, translated_content = translate_content("unexpected input")
        self.assertEqual(is_english, True)
        self.assertEqual(translated_content, "unexpected input")
        
if __name__ == "__main__":
    unittest.main()