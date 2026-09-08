from parlamonitor.emtsv import Token
from parlamonitor.propernouns import PROPER_NOUN_LEMMAS, repair_sentences, repair_token

# Exactly what emtsv returns for "Taps a Jobbik soraiban."
JOBBIK = Token("Jobbik", "jó", "[/Adj][_Comp/Adj][_Design/Adj][Nom]")


def test_the_party_name_is_restored():
    fixed, repaired = repair_token(JOBBIK)
    assert (fixed.lemma, fixed.xpostag, repaired) == ("Jobbik", "[/N][Nom]", True)


def test_the_surface_form_is_never_rewritten():
    assert repair_token(JOBBIK)[0].form == "Jobbik"


def test_the_lowercase_adjective_is_left_alone():
    # "jobbik" the comparative really does lemmatise to "jó"; 5 occurrences.
    token = Token("jobbik", "jó", "[/Adj][_Comp/Adj][_Design/Adj][Nom]")
    assert repair_token(token) == (token, False)


def test_names_emmorph_already_gets_right_are_untouched():
    for form in ("Fidesz", "KDNP", "MSZP", "LMP", "DK"):
        token = Token(form, form, "[/N][Nom]")
        assert repair_token(token) == (token, False)


def test_repairing_an_already_correct_token_is_not_counted_as_a_repair():
    token = Token("Jobbik", "Jobbik", "[/N][Nom]")
    assert repair_token(token) == (token, False)


def test_wsafter_survives_the_repair():
    assert (
        repair_token(Token("Jobbik", "jó", "[/Adj]", wsafter="\n"))[0].wsafter == "\n"
    )


def test_sentences_are_repaired_and_counted():
    sentences = [
        [Token("Taps", "taps", "[/N][Nom]"), JOBBIK],
        [Token("Momentum", "momentum", "[/N][Nom]")],
    ]
    repaired, n = repair_sentences(sentences)
    assert [[t.lemma for t in s] for s in repaired] == [
        ["taps", "Jobbik"],
        ["Momentum"],
    ]
    assert n == 2


def test_every_entry_maps_a_capitalised_form():
    # The rule only fires on capitalised forms; a lowercase key could never be
    # reached without also catching the common word it is homographic with.
    assert all(form[:1].isupper() for form in PROPER_NOUN_LEMMAS)
