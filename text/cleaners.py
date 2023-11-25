""" from https://github.com/keithito/tacotron """

"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import re
from unidecode import unidecode
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
backend = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True)


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


# List of (regular expression, replacement) pairs for abbreviations:
_map_unk_words_vi = [
    (re.compile("\\b%s\\b" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("taxi", "tắc xi"),
        ("you", "diu"),
        ("alo", "a lô"),
        ("boss", "bót"),
        ("shop", "sốp"),
        ("centimet", "sen ti mét"),
        ("atm", "ây ti em"),
        ("scandal", "sờ can đồ"),
        ("tivi", "ti vi"),
        ("oke", "ô kê"),
        ("toilet", "toi lét"),
        ("sofa", "sô pha"),
        ("studio", "sờ tu đi ô"),
        ("mode", "mốt"),
        ("venice", "vơ nít"),
        ("ok", "ô kê"),
        ("sexy", "sếch xi"),
        ("honey", "hơ ni"),
        ("love", "lớp"),
        ("jean", "din"),
        ("casting", "cát sờ ting"),
    ]
]


def expand_unk_words_vi(text):
    for regex, replacement in _map_unk_words_vi:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """Pipeline for English text, including abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(text, language="en-us", backend="espeak", strip=True)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(
        text,
        language="en-us",
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
    )
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def english_cleaners3(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = backend.phonemize([text], strip=True)[0]
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def vietnamese_cleaner2(text):
    """Modify version of english_cleaner2"""
    text = lowercase(text)
    text = text.replace("_", " ")
    print(text)
    text = expand_unk_words_vi(text)
    phonemes = phonemize(
        text,
        language="vi",
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
        language_switch='remove-utterance',
    )
    phonemes = collapse_whitespace(phonemes)
    return phonemes
