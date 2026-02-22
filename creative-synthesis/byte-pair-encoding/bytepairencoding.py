from collections import Counter, defaultdict
from itertools import pairwise

from pretoken import Pretoken

class BytePairEncoder:
    def __init__(self):
        self.trained = False
        self.vocabulary: list[str] = []
    
    @staticmethod
    def _normalize(text: str) -> str:
        """
        Does nothing for now.
        
        :param text: The text to normalize. 
        :type text: str
        :return: The normalized text. 
        :rtype: str
        """
        return text

    @staticmethod
    def _pretokenize(text: str) -> list[Pretoken]:
        """
        Pretokenize some text. 
        
        :param text: The text to pretokenize. 
        :type text: str
        :return: The list of Pretokens. 
        :rtype: list[Pretoken]
        """
        # TODO: make it split on whitespace and punctuation and preserve punc.
        pretokens_text = text.split()
        pretokens: list[Pretoken] = []

        for pretoken in pretokens_text:
            pretokens.append(Pretoken(pretoken))

        return pretokens
    

    @staticmethod
    def _generate_token_corpus(pretokens: list[Pretoken]) -> list[tuple[Pretoken, int]]:
        """
        Generate a corpus of tokens and their number of occurrences from a list of Pretokens. 
        
        :param pretokens: The pretokenized text to generate a corpus from.
        :type pretokens: list[Pretoken]
        :return: A list of pretokens and the number of occurrences.
        :rtype: list[tuple[Pretoken, int]]
        """
        counter = Counter(pretokens)

        return [entry for entry in counter.most_common() if entry[1] != 0]
        

    def train(self, text: str, vocabulary_size: int = -1):
        """
        Train the Byte-Pair Encoding tokenizer on a sample of text, to a specified vocabulary size.
        
        :param text: The text to train the tokenizer on.
        :type text: str
        :param vocabulary_size: The maximum vocabulary size. A value of -1 will auto-detect the vocabulary size using the XXX algorithm (todo). 
        :type vocabulary_size: int
        """
        pretokens = self._pretokenize(text)
        corpus = self._generate_token_corpus(pretokens)

        # set initial vocabulary
        vocabulary: set[str] = set()
        for pretoken, _ in corpus:
            vocabulary.update(pretoken.get_tokens()) # all characters
        self.vocabulary = list(vocabulary)

        while len(self.vocabulary) < vocabulary_size:
            token_pair_counts: defaultdict[str, int] = defaultdict(int)
            for pretoken, count in corpus:
                for t1, t2 in pairwise(pretoken.get_tokens()):
                    token_pair_counts[t1 + t2] += count

            # we can't make any more tokens, so we break early, even though we haven't reached the maximum vocabulary size
            if len(token_pair_counts) == 0: break

            new_token = max(token_pair_counts.items(), key=lambda i: i[1])[0]
            self.vocabulary.append(new_token)

            for pretoken, _ in corpus:
                pretoken.apply_merge_rule(new_token)


        self.trained = True


    def tokenize(self, text: str) -> list[str]:
        """
        Use the trained tokenizer to tokenize some text. 
        
        :param text: The text to tokenize.
        :type text: str
        """
        if not self.trained:
            raise Exception("BytePairEncoder has not been trained yet")
        
        pretokens = self._pretokenize(text)

        for pretoken in pretokens:
            for merge_rule in self.vocabulary:
                pretoken.apply_merge_rule(merge_rule)

        tokens: list[str] = []

        for pretoken in pretokens:
            tokens.extend(pretoken.get_tokens())

        return tokens
