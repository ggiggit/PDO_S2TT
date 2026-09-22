from pdo_s2tt.evaluation import units as evaluation_units
from pdo_s2tt.simultaneous_metrics import words
from pdo_s2tt.training.reward import units as reward_units


def test_japanese_units_include_han_hiragana_katakana_and_extensions():
    text = "漢あアㇰ ABC élève"
    expected = ("漢", "あ", "ア", "ㇰ", "abc", "élève")

    assert tuple(words(text)) == expected
    assert evaluation_units(text, "ja") == expected
    assert reward_units(text, "ja") == expected


def test_latin_extended_text_is_kept_as_words_for_laal():
    assert words("élève déjà") == ["élève", "déjà"]
