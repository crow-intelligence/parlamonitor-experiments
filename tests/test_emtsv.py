import pytest
import requests

from parlamonitor.emtsv import (
    AlignmentError,
    EmtsvError,
    Token,
    analyse,
    analyse_lines,
    decode_wsafter,
    is_content_word,
    lemmatize,
    parse_tsv,
    parse_tsv_sentences,
    parse_xpostag,
    split_by_input_line,
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


# --- sentence boundaries and batched, line-aligned analysis -----------------


def test_wsafter_is_decoded_from_its_escaped_literal():
    assert decode_wsafter('" "') == " "
    assert decode_wsafter('"\\n"') == "\n"
    assert decode_wsafter('""') == ""


def test_an_unparseable_wsafter_cell_loses_its_quotes_rather_than_raising():
    assert decode_wsafter('"') == ""
    assert decode_wsafter("nonsense") == "nonsense"


def test_sentences_are_split_at_the_blank_lines():
    sentences = parse_tsv_sentences(REAL_RESPONSE)
    assert [len(s) for s in sentences] == [6, 1]
    assert sentences[1][0].lemma == "parlament"


def test_parse_tsv_is_the_flattening_of_parse_tsv_sentences():
    flat = parse_tsv(REAL_RESPONSE)
    nested = parse_tsv_sentences(REAL_RESPONSE)
    assert flat == [token for sentence in nested for token in sentence]


def test_a_chain_without_tok_still_parses_and_leaves_wsafter_empty():
    payload = "form\tlemma\txpostag\nTaps\ttaps\t[/N][Nom]\n"
    (token,) = parse_tsv(payload)
    assert token.wsafter == ""


# --- regrouping a batched response onto its input lines ---------------------


def line_ending(form, lemma="x", tag="[/N]"):
    return Token(form, lemma, tag, wsafter="\n")


def test_newlines_in_wsafter_mark_the_input_line_boundaries():
    sentences = [
        [Token("Taps", "taps", "[/N]"), line_ending(".", ".", "[Punct]")],
        [line_ending("Csenget", "cseng", "[/V]")],
    ]
    grouped = split_by_input_line(sentences, 2)
    assert [[[t.lemma for t in s] for s in line] for line in grouped] == [
        [["taps", "."]],
        [["cseng"]],
    ]


def test_several_sentences_can_belong_to_one_input_line():
    sentences = [
        [Token("Taps", "taps", "[/N]"), Token(".", ".", "[Punct]", wsafter=" ")],
        [Token("Nagy", "nagy", "[/Adj]"), line_ending("taps", "taps", "[/N]")],
    ]
    (line,) = split_by_input_line(sentences, 1)
    assert [[t.lemma for t in s] for s in line] == [["taps", "."], ["nagy", "taps"]]


def test_a_newline_inside_a_sentence_still_ends_the_line():
    # tok should never do this, but merging two input lines into one sentence
    # would corrupt every collocation drawn from them.
    sentences = [[line_ending("a"), Token("b", "b", "[/N]", wsafter="\n")]]
    grouped = split_by_input_line(sentences, 2)
    assert [[[t.lemma for t in s] for s in line] for line in grouped] == [
        [["x"]],
        [["b"]],
    ]


def test_a_response_that_does_not_account_for_every_line_is_refused():
    with pytest.raises(AlignmentError, match="expected 3"):
        split_by_input_line([[line_ending("a")]], 3)


# --- analyse_lines ----------------------------------------------------------

BATCH_RESPONSE = (
    "form\twsafter\tlemma\txpostag\n"
    'Taps\t""\ttaps\t[/N][Nom]\n'
    '.\t"\\n"\t.\t[Punct]\n'
    "\n"
    'Csenget\t""\tcseng\t[/V][Prs.NDef.3Sg]\n'
    '.\t"\\n"\t.\t[Punct]\n'
)


def test_a_batch_is_sent_as_one_request_and_split_back_apart():
    session = FakeSession(FakeResponse(BATCH_RESPONSE))
    result = analyse_lines(["Taps.", "Csenget."], session=session)
    assert len(session.calls) == 1
    assert [[[t.lemma for t in s] for s in line] for line in result] == [
        [["taps", "."]],
        [["cseng", "."]],
    ]


def test_the_request_body_is_newline_terminated_so_the_last_line_has_a_boundary():
    session = FakeSession(FakeResponse(BATCH_RESPONSE))
    analyse_lines(["Taps.", "Csenget."], session=session)
    sent = session.calls[0][1]["files"]["text"][1]
    assert sent == "Taps.\nCsenget.\n"


# One input line, so it aligns whatever single-line batch it answers.
ONE_LINE_RESPONSE = 'form\twsafter\tlemma\txpostag\nX\t"\\n"\tx\t[/N][Nom]\n'


def test_the_batch_caps_split_the_work():
    session = FakeSession(FakeResponse(ONE_LINE_RESPONSE))
    analyse_lines(["X", "X"], batch_lines=1, session=session)
    assert len(session.calls) == 2


def test_the_word_cap_also_closes_a_batch():
    session = FakeSession(FakeResponse(ONE_LINE_RESPONSE))
    analyse_lines(["X", "X"], batch_words=1, session=session)
    assert len(session.calls) == 2


def test_a_line_longer_than_the_whole_budget_is_still_sent():
    long_line = (
        "form\twsafter\tlemma\txpostag\n"
        'a\t" "\ta\t[/N][Nom]\n'
        'b\t" "\tb\t[/N][Nom]\n'
        'c\t"\\n"\tc\t[/N][Nom]\n'
    )
    session = FakeSession(FakeResponse(long_line))
    analyse_lines(["a b c"], batch_words=2, session=session)
    assert len(session.calls) == 1


@pytest.mark.parametrize(("lines", "words"), [(0, 10), (10, 0)])
def test_a_batch_cap_below_one_is_refused(lines, words):
    with pytest.raises(ValueError, match="at least 1"):
        analyse_lines(["Taps."], batch_lines=lines, batch_words=words)


def test_a_misaligned_batch_falls_back_to_one_request_per_line():
    # The response accounts for one input line, but two were sent.
    short = (
        "form\twsafter\tlemma\txpostag\n"
        'Taps\t""\ttaps\t[/N][Nom]\n'
        '.\t"\\n"\t.\t[Punct]\n'
    )
    single = (
        'form\twsafter\tlemma\txpostag\nX\t""\tx\t[/N][Nom]\n.\t"\\n"\t.\t[Punct]\n'
    )
    session = FakeSession(FakeResponse(short), FakeResponse(single))
    result = analyse_lines(["Taps.", "Csenget."], session=session)
    # One failed batch, then one request per line.
    assert len(session.calls) == 3
    assert [[[t.lemma for t in s] for s in line] for line in result] == [
        [["x", "."]],
        [["x", "."]],
    ]


def test_a_response_that_cannot_rebuild_the_request_body_is_not_trusted():
    # Correct line count, but the tokens do not reconstruct what was sent, so
    # the mapping is unverifiable and the batch must not be used.
    mangled = (
        "form\twsafter\tlemma\txpostag\n"
        'Taps\t""\ttaps\t[/N][Nom]\n'
        '.\t"\\n"\t.\t[Punct]\n'
        'Csenget\t""\tcseng\t[/V]\n'
        '?\t"\\n"\t?\t[Punct]\n'
    )
    session = FakeSession(FakeResponse(mangled))
    analyse_lines(["Taps.", "Csenget."], session=session)
    assert len(session.calls) == 3  # the batch, then each line on its own


def test_a_line_that_fails_even_on_its_own_raises():
    session = FakeSession(FakeResponse("", status=500))
    with pytest.raises(EmtsvError):
        analyse_lines(["Taps."], retries=1, session=session)


def test_blank_lines_analyse_to_nothing_without_a_request():
    session = FakeSession(FakeResponse(BATCH_RESPONSE))
    assert analyse_lines(["   ", ""], session=session) == [[], []]
    assert session.calls == []


def test_progress_is_reported_per_batch():
    session = FakeSession(FakeResponse(BATCH_RESPONSE))
    seen = []
    analyse_lines(
        ["Taps.", "Csenget."],
        batch_lines=1,
        session=session,
        on_batch=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 2), (2, 2)]
