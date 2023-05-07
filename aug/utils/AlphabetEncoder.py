import string
from typing import List


class AlphabetEncoder:

    def __init__(self):
        self.alphabet = list("()\"äåö,!?.:'~")  # lower case, digits, special chars: äåö,!"'
        self.alphabet.extend(string.digits)
        self.alphabet.extend(string.ascii_lowercase)
        self.alphabet.sort()
        self.alphabet.insert(0, "#")  # blank
        self.alphabet.append(" ")  # space

    def encode(self, input: str) -> List[int]:
        return [self.alphabet.index(c) for c in input]

    def decode(self, input: List[int]) -> str:
        return "".join([self.alphabet[c] for c in input if c != 0])

    def alphabetSize(self) -> int:
        return len(self.alphabet)
