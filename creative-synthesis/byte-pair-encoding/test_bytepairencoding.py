from bytepairencoding import BytePairEncoder
from pretoken import Pretoken

def _get_text(word_counts):
    input_text = ""
    for word, count in word_counts:
        input_text += f"{word} " * count

    return input_text

def test_init_BPE():
    bpe = BytePairEncoder()
    assert bpe, "No BytePairEncoder object"

def test_tokenize_no_vocabulary():
    input_text = "hug pugs"

    bpe = BytePairEncoder()
    bpe.trained = True
    bpe.vocabulary = list(set(input_text))

    result = bpe.tokenize(input_text)
    assert result == list("hugpugs")

def test_tokenize_sample_vocabulary():
    input_text = "hug pugs"

    bpe = BytePairEncoder()
    bpe.trained = True
    bpe.vocabulary = list(set(input_text)) + ["ug"]

    result = bpe.tokenize(input_text)

    assert result == ['h', 'ug', 'p', 'ug', 's']
    
def test_corpus_generator():
    bpe = BytePairEncoder()

    word_counts = [("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)]
    input_text = _get_text(word_counts)

    pretokens = bpe._pretokenize(input_text)
    corpus = bpe._generate_token_corpus(pretokens)

    assert corpus == sorted([(Pretoken(w), c) for (w, c) in word_counts], key=lambda e:e[1], reverse=True)

def test_train_vocabulary():
    word_counts = [("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)]
    input_text = _get_text(word_counts)

    bpe = BytePairEncoder()
    bpe.train(input_text, 10)
    # we make no guarantees about the order of tokens
    assert sorted(bpe.vocabulary) == sorted(["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"])
    
def test_train_and_tokenize():
    word_counts = [("hug", 5), ("pug", 3), ("pun", 6), ("bun", 2), ("hugs", 3)]
    input_text = _get_text(word_counts)

    bpe = BytePairEncoder()
    bpe.train(input_text, 10)

    tokenized_text = bpe.tokenize(input_text)

    # probably a better way to test this
    assert tokenized_text == ['hug', 'hug', 'hug', 'hug', 'hug', 'p', 'ug', 'p', 'ug', 'p', 'ug', 'p', 'un', 'p', 'un', 'p', 'un', 'p', 'un', 'p', 'un', 'p', 'un', 'b', 'un', 'b', 'un', 'hug', 's', 'hug', 's', 'hug', 's']

def test_not_reach_max_tokens():
    input_text = "huggs"
    bpe = BytePairEncoder()
    bpe.train(input_text, 10)
    tokenized_text = bpe.tokenize(input_text)

    assert bpe.vocabulary[-1] == input_text
    assert tokenized_text == [input_text]
