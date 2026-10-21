import unittest
from pathlib import Path

from extension.lib import wordlist

WORDS_PATH = Path(__file__).resolve().parents[1] / "data" / "american-english.txt"


class TestCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.words = wordlist.load_words(WORDS_PATH)

    def test_word_based_label_is_covered(self):
        self.assertGreaterEqual(wordlist.coverage("doeslimitingfreeprogram", self.words), 0.8)

    def test_random_label_is_not_covered(self):
        self.assertLessEqual(wordlist.coverage("hdkdpdsrvn", self.words), 0.4)

    def test_empty_and_non_letter_labels(self):
        self.assertEqual(wordlist.coverage("", self.words), 0.0)
        self.assertEqual(wordlist.coverage("1234-5", self.words), 0.0)
        value = wordlist.coverage("xn--80ak6aa92e", self.words)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_words_are_lowercase_letters_of_min_length(self):
        self.assertTrue(all(w.isalpha() and w.islower() and len(w) >= 3 for w in self.words))


class TestMainLabel(unittest.TestCase):
    def test_main_label(self):
        self.assertEqual(wordlist.main_label("baseballpark.co.kr"), "baseballpark")
        self.assertEqual(wordlist.main_label("hdkdpds9r3v4n.net"), "hdkdpds9r3v4n")
        self.assertEqual(wordlist.main_label("localhost"), "localhost")


if __name__ == "__main__":
    unittest.main()
