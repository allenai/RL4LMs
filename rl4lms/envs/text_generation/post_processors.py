from nltk.tokenize import sent_tokenize


def three_sentence_summary(text):
    """
    Returns first three sentences from the generated text
    """
    return "\n".join(sent_tokenize(text)[:3])
