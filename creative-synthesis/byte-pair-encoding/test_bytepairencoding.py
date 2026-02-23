from bytepairencoding import BytePairEncoder
from pretoken import Pretoken

def _get_text(word_counts):
    input_text = ""
    for word, count in word_counts:
        input_text += f"{word} " * count

    return input_text[:-1] # leave out the trailing whitespace

def test_init_BPE():
    bpe = BytePairEncoder()
    assert bpe, "No BytePairEncoder object"

def test_tokenize_no_vocabulary():
    input_text = "hug pugs"

    bpe = BytePairEncoder()
    bpe.trained = True
    bpe.vocabulary = list(set(input_text))

    result = bpe.tokenize(input_text)
    assert result == list("hugĠpugs")

def test_tokenize_sample_vocabulary():
    input_text = "hug pugs"

    bpe = BytePairEncoder()
    bpe.trained = True
    bpe.vocabulary = list(set(input_text)) + ["ug"]

    result = bpe.tokenize(input_text)

    print(result)
    assert result == ['h', 'ug', 'Ġ', 'p', 'ug', 's']
    
def test_corpus_generator():
    bpe = BytePairEncoder()

    word_counts = [("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)]
    input_text = _get_text(word_counts)

    pretokens = bpe._pretokenize(input_text)
    corpus = bpe._generate_token_corpus(pretokens)

    assert corpus[0] == (Pretoken("Ġpun"), 12)
    # could check others but I'm lazy
    assert corpus[5] == (Pretoken("hug"), 1) # first word has no leading space

def test_train_vocabulary():
    word_counts = [("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)]
    input_text = _get_text(word_counts)

    bpe = BytePairEncoder()
    bpe.train(input_text, 12)
    print(bpe.vocabulary)
    # we make no guarantees about the order of tokens
    assert sorted(bpe.vocabulary) == sorted(['p', 'b', 'u', 'h', 'Ġ', 'n', 'g', 's', 'ug', 'Ġp', 'un', 'hug'])
    
def test_train_and_tokenize():
    word_counts = [("hug", 5), ("pug", 3), ("pun", 6), ("bun", 2), ("hugs", 3)]
    input_text = _get_text(word_counts)

    bpe = BytePairEncoder()
    bpe.train(input_text, 14)

    tokenized_text = bpe.tokenize(input_text)

    assert tokenized_text == ['hug', 'Ġhug', 'Ġhug', 'Ġhug', 'Ġhug', 'Ġp', 'ug', 'Ġp', 'ug', 'Ġp', 'ug', 'Ġpun', 'Ġpun', 'Ġpun', 'Ġpun', 'Ġpun', 'Ġpun', 'Ġ', 'b', 'un', 'Ġ', 'b', 'un', 'Ġhug', 's', 'Ġhug', 's', 'Ġhug', 's']

def test_not_reach_max_tokens():
    input_text = "huggs"
    bpe = BytePairEncoder()
    bpe.train(input_text, 10)
    tokenized_text = bpe.tokenize(input_text)

    assert bpe.vocabulary[-1] == input_text
    assert tokenized_text == [input_text]

def test_tokenize_punctuation():
    input_text = "a@b.c:d&e"
    bpe = BytePairEncoder()
    bpe.train(input_text)
    tokenized_text = bpe.tokenize(input_text)
    
    assert tokenized_text == list(input_text)