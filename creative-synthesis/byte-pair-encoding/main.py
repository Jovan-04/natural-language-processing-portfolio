from bytepairencoding import BytePairEncoder

def main():
    word_counts = [("hug", 5), ("pug", 3), ("pun", 6), ("bun", 2), ("hugs", 3)]
    input_text = ""
    for word, count in word_counts:
        input_text += f"{word} " * count

    bpe = BytePairEncoder()
    bpe.train(input_text, 10)

    tokenized_text = bpe.tokenize(input_text)
    print("input text:")
    print(input_text)
    print("tokenized text:")
    print(tokenized_text)
    print("vocabulary:")
    print(bpe.vocabulary)


if __name__ == "__main__":
    main()
