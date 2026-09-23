from src.tokenizers.word_tokenizer import WordTokenizer


def test_word_tokenizer_builds_vocabulary():
    texts = [
        "I love machine learning.",
        "I love deep learning.",
    ]

    tokenizer = WordTokenizer()
    tokenizer.fit(texts)

    assert tokenizer.is_fitted
    assert tokenizer.vocab_size > 4

    assert "machine" in tokenizer.get_vocab()
    assert "learning" in tokenizer.get_vocab()


def test_word_tokenizer_encode_decode():
    texts = [
        "I love machine learning."
    ]

    tokenizer = WordTokenizer()
    tokenizer.fit(texts)

    token_ids = tokenizer.encode(
        "I love machine learning."
    )

    decoded = tokenizer.decode(token_ids)

    assert decoded == "I love machine learning."


def test_word_tokenizer_handles_oov():
    train = [
        "I love machine learning."
    ]

    test = [
        "I love quantum computing."
    ]

    tokenizer = WordTokenizer()
    tokenizer.fit(train)

    token_ids = tokenizer.encode(test[0])

    unk_id = tokenizer.token_id("<UNK>")

    assert unk_id in token_ids

    assert tokenizer.oov_rate(test) > 0.0


def test_word_tokenizer_special_tokens():
    tokenizer = WordTokenizer()

    tokenizer.fit(["hello world"])

    token_ids = tokenizer.encode(
        "hello world",
        add_special_tokens=True,
    )

    assert token_ids[0] == tokenizer.token_id("<BOS>")
    assert token_ids[-1] == tokenizer.token_id("<EOS>")