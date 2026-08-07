import pytest
import requests

from parlamonitor.emtsv import (
    EmtsvError,
    Token,
    analyse,
    is_content_word,
    lemmatize,
    parse_tsv,
    parse_xpostag,
)

# Trimmed from a real response of `POST /tok/morph/pos` for
# "A kormány benyújtotta a törvényjavaslatot." -- the anas column is truncated
# here, but its position and the tab layout are exactly as emtsv sends them.
REAL_RESPONSE = (
    "form\twsafter\tanas\tlemma\txpostag\n"
    'A\t" "\t[{"lemma": "a", "tag": "[/Det|Art.Def]"}]\ta\t[/Det|Art.Def]\n'
    'kormány\t" "\t[{"lemma": "kormány", "tag": "[/N][Nom]"}]\tkormány\t[/N][Nom]\n'
    'benyújtotta\t" "\t[{"lemma": "benyújt"}]\tbenyújt\t[/V][Pst.Def.3Sg]\n'
    'a\t" "\t[{"lemma": "a"}]\ta\t[/Det|Art.Def]\n'
    'törvényjavaslatot\t""\t[{"lemma": "törvényjavaslat"}]'
    "\ttörvényjavaslat\t[/N][Acc]\n"
    '.\t"\\n"\t[]\t.\t[Punct]\n'
    "\n"
    'Parlamentben\t""\t[{"lemma": "parlament"}]\tparlament\t[/N][Ine]\n'
)


class FakeResponse:
    """Minimal stand-in for a requests response, holding raw utf-8 bytes."""

    def __init__(self, body="", status=200):
        self.content = body.encode("utf-8")
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class FakeSession:
    """Records calls and replays a queue of responses or exceptions."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        result = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(result, Exception):
            raise result
        return result


# --- tag parsing ------------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("[/N][Nom]", ("N", ())),
        ("[/V][Pst.Def.3Sg]", ("V", ())),
        ("[/Adj|nat][Nom]", ("Adj", ("nat",))),
        ("[/N|Pro|(Post)][Nom]", ("N", ("Pro", "(Post)"))),
        ("[/Det|Art.Def]", ("Det", ("Art.Def",))),
        ("[Punct]", (None, ())),
        ("", (None, ())),
        ("[/N", (None, ())),  # unterminated
    ],
)
def test_parse_xpostag(tag, expected):
    assert parse_xpostag(tag) == expected


@pytest.mark.parametrize(
    ("tag", "keep"),
    [
        ("[/N][Nom]", True),
        ("[/N][Ine]", True),
        ("[/V][Pst.Def.3Sg]", True),
        ("[/Adj][Nom]", True),
        ("[/Adv]", True),
        ("[/Det|Art.Def]", False),
        ("[/Cnj]", False),
        ("[/Num][Nom]", False),
        ("[Punct]", False),
        ("[/N|Pro|(Post)][Nom]", False),  # pronoun, tagged under N
    ],
)
def test_is_content_word(tag, keep):
    assert is_content_word(tag) is keep


def test_pronouns_kept_when_asked():
    assert is_content_word("[/N|Pro|(Post)][Nom]", drop_pronouns=False) is True


def test_keep_categories_are_configurable():
    assert is_content_word("[/N][Nom]", keep=("V",)) is False


# --- TSV parsing ------------------------------------------------------------


def test_parse_tsv_reads_real_response():
    tokens = parse_tsv(REAL_RESPONSE)
    assert [t.form for t in tokens][:3] == ["A", "kormány", "benyújtotta"]
    assert [t.lemma for t in tokens][:3] == ["a", "kormány", "benyújt"]
    assert tokens[2].xpostag == "[/V][Pst.Def.3Sg]"


def test_parse_tsv_skips_sentence_separators():
    tokens = parse_tsv(REAL_RESPONSE)
    # The blank line before Parlamentben must not become a token.
    assert all(t.form for t in tokens)
    assert tokens[-1] == Token("Parlamentben", "parlament", "[/N][Ine]")


def test_parse_tsv_locates_columns_by_name_not_position():
    reordered = "xpostag\tlemma\tform\n[/N][Ine]\tparlament\tParlamentben\n"
    assert parse_tsv(reordered) == [Token("Parlamentben", "parlament", "[/N][Ine]")]


def test_parse_tsv_without_lemma_column_raises():
    """A `tok`-only chain produces no lemma; that must not pass silently."""
    with pytest.raises(EmtsvError, match="lemma"):
        parse_tsv('form\twsafter\nkormány\t" "\n')


def test_parse_tsv_empty_raises():
    with pytest.raises(EmtsvError, match="empty"):
        parse_tsv("   \n\n")


# --- the client -------------------------------------------------------------


def test_analyse_posts_to_the_module_chain_path():
    session = FakeSession(FakeResponse(REAL_RESPONSE))
    analyse("A kormány.", session=session, base_url="http://host:5000/")

    url, kwargs = session.calls[0]
    assert url == "http://host:5000/tok/morph/pos"
    # multipart `text` field, matching `curl -F 'text=...'`
    assert kwargs["files"] == {"text": (None, "A kormány.")}


def test_analyse_decodes_utf8_not_latin1():
    """The requests library defaults to ISO-8859-1 when no charset is sent."""
    session = FakeSession(FakeResponse(REAL_RESPONSE))
    tokens = analyse("bármi", session=session)
    assert "törvényjavaslat" in [t.lemma for t in tokens]


def test_analyse_retries_then_succeeds():
    session = FakeSession(
        requests.ConnectionError("refused"), FakeResponse(REAL_RESPONSE)
    )
    tokens = analyse("A kormány.", session=session, retries=3, backoff=0)
    assert len(session.calls) == 2
    assert tokens


def test_analyse_never_falls_back_to_the_input_text():
    """The spec's raw-text fallback would hide a total service failure."""
    text = "A kormány benyújtotta a törvényjavaslatot."
    session = FakeSession(FakeResponse("", status=500))

    with pytest.raises(EmtsvError) as excinfo:
        analyse(text, session=session, retries=2, backoff=0)

    assert len(session.calls) == 2
    assert text not in str(excinfo.value)


def test_analyse_empty_text_raises():
    with pytest.raises(ValueError, match="empty"):
        analyse("   \n ", session=FakeSession(FakeResponse(REAL_RESPONSE)))


def test_analyse_rejects_zero_retries():
    with pytest.raises(ValueError, match="at least 1"):
        analyse("x", retries=0, session=FakeSession(FakeResponse(REAL_RESPONSE)))


# --- lemmatize --------------------------------------------------------------


def test_lemmatize_keeps_content_words_only():
    session = FakeSession(FakeResponse(REAL_RESPONSE))
    assert lemmatize("bármi", session=session) == [
        "kormány",
        "benyújt",
        "törvényjavaslat",
        "parlament",
    ]


def test_lemmatize_can_keep_everything():
    session = FakeSession(FakeResponse(REAL_RESPONSE))
    lemmas = lemmatize("bármi", content_only=False, session=session)
    assert lemmas[:2] == ["a", "kormány"]
    assert "." in lemmas
