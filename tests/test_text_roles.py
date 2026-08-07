import pytest
from hypothesis import given
from hypothesis import strategies as st

from parlamonitor.roles import (
    DEBATE,
    QA_ROLES,
    ROLE_BY_SPEECH_TYPE,
    compare_role_sources,
    discourse_role,
    is_qa,
    qa_turn_uids,
)
from parlamonitor.stopwords import (
    LIGHT_VERBS,
    PARLIAMENTARY,
    SPACY_HU,
    hungarian_stopwords,
)
from parlamonitor.text import strip_salutation

# Openings taken verbatim from cycle-43 speeches.
REAL_OPENINGS = [
    "Tisztelt Országgyűlés! Tájékoztatom önöket, hogy minden képviselő",
    "Tisztelt Elnök Asszony! Tisztelt Képviselőtársaim! Napra pontosan",
    "Tisztelt Köztársasági Elnök Úr! Tisztelt Országgyűlés! "
    "Tisztelt Hölgyeim és Uraim! A válasz egyszerű.",
]


# --- salutation stripping ---------------------------------------------------


@pytest.mark.parametrize("text", REAL_OPENINGS)
def test_strip_salutation_removes_real_openings(text):
    stripped, removed = strip_salutation(text)
    assert removed > 0
    assert not stripped.lower().startswith("tisztelt")
    assert stripped.strip()


def test_strip_salutation_handles_koszonom_then_tisztelt():
    text = "Köszönöm a szót. Tisztelt Ház! Az egészségügyről szólnék."
    assert strip_salutation(text) == ("Az egészségügyről szólnék.", 5)


def test_strip_salutation_leaves_substantive_text_alone():
    text = "A kormány benyújtotta a költségvetési törvényjavaslatot."
    assert strip_salutation(text) == (text, 0)


def test_strip_salutation_never_empties_a_speech():
    """One cycle-43 speech is a salutation and nothing else."""
    text = "Tisztelt Elnök Úr! Tisztelt Képviselőtársaim!"
    assert strip_salutation(text) == (text, 0)


def test_strip_salutation_refuses_to_eat_too_much():
    text = "Tisztelt Ház, " + " ".join(f"szó{i}" for i in range(100)) + ". Utána."
    assert strip_salutation(text, max_words=60) == (text, 0)


@given(st.text(max_size=300))
def test_strip_salutation_is_idempotent(text):
    once, _ = strip_salutation(text)
    twice, second_removed = strip_salutation(once)
    assert twice == once
    assert second_removed == 0


@given(st.text(max_size=300))
def test_strip_salutation_never_adds_words(text):
    stripped, removed = strip_salutation(text)
    assert len(stripped.split()) + removed <= len(text.split())
    assert removed >= 0


# --- discourse roles --------------------------------------------------------


@pytest.mark.parametrize(
    ("speech_type", "role"),
    [
        ("elhangzik az interpelláció/kérdés/azonnali kérdés", "question"),
        ("kérdés megválaszolva", "answer"),
        ("interpelláció szóban megválaszolva", "answer"),
        ("azonnali kérdésre adott képviselői viszonválasz", "mp_rejoinder"),
        ("azonnali kérdésre adott miniszteri viszonválasz", "minister_rejoinder"),
        ("képviselő elfogadta a választ", "reaction"),
        ("képviselő elutasította a választ", "reaction"),
        ("vezérszónoki felszólalás", DEBATE),
        ("kétperces felszólalás", DEBATE),
        ("valami teljesen új típus", DEBATE),
    ],
)
def test_discourse_role(speech_type, role):
    assert discourse_role({"speech_type": speech_type}) == role


def test_discourse_role_missing_key_is_debate():
    assert discourse_role({}) == DEBATE


def test_is_qa_matches_the_role_set():
    for speech_type, role in ROLE_BY_SPEECH_TYPE.items():
        assert is_qa({"speech_type": speech_type}) is (role in QA_ROLES)
    assert is_qa({"speech_type": "felszólalás"}) is False


def test_qa_turn_uids_skips_missing_and_null():
    records = [
        {"turns": [{"uid": "a"}, {"uid": None}, {}]},
        {"turns": None},
        {},
        {"turns": [{"uid": "b"}, {"uid": "a"}]},
    ]
    assert qa_turn_uids(records) == {"a", "b"}


def test_compare_role_sources_agrees_when_consistent():
    speeches = [
        {"uid": "a", "speech_type": "kérdés megválaszolva"},
        {"uid": "b", "speech_type": "felszólalás"},
    ]
    result = compare_role_sources(speeches, {"a"})
    assert result["agree"] is True
    assert result["by_role"] == {"answer": 1, DEBATE: 1}
    assert result["n_qa_uids_in_speeches"] == 1


def test_compare_role_sources_reports_disagreement():
    speeches = [
        {"uid": "a", "speech_type": "kérdés megválaszolva"},
        {"uid": "b", "speech_type": "felszólalás"},
    ]
    # 'b' is in the Q&A file but typed as debate; 'a' is typed Q&A but absent.
    result = compare_role_sources(speeches, {"b"})
    assert result["agree"] is False
    assert result["typed_qa_not_in_qa_file"] == ["a"]
    assert result["in_qa_file_not_typed_qa"] == ["b"]


def test_compare_role_sources_ignores_qa_uids_absent_from_speeches():
    """37 of 659 Q&A turns have no published speech; that is not a mismatch."""
    speeches = [{"uid": "a", "speech_type": "kérdés megválaszolva"}]
    result = compare_role_sources(speeches, {"a", "unpublished"})
    assert result["agree"] is True
    assert result["n_qa_uids_in_speeches"] == 1


# --- stopwords --------------------------------------------------------------


def test_stopwords_contain_the_verbs_the_spec_asked_for():
    stops = hungarian_stopwords()
    assert {"tud", "tesz", "benyújt"} <= stops


def test_stopwords_keep_speech_act_verbs():
    """szavaz/elutasít/módosít distinguish what a speech does; keep them."""
    stops = hungarian_stopwords()
    assert not ({"szavaz", "elutasít", "elfogad", "módosít", "támogat"} & stops)


def test_stopwords_keep_topical_nouns():
    stops = hungarian_stopwords()
    assert not ({"költségvetés", "egészségügy", "migráció", "elnök"} & stops)


def test_light_verbs_cover_what_spacy_misses():
    """The reason LIGHT_VERBS exists: spaCy has van/kell/lesz but not these."""
    missing_from_spacy = {"tud", "mond", "beszél", "gondol"}
    assert not (missing_from_spacy & SPACY_HU)
    assert missing_from_spacy <= LIGHT_VERBS


def test_hungarian_stopwords_is_the_union():
    assert hungarian_stopwords() == SPACY_HU | LIGHT_VERBS | PARLIAMENTARY


@pytest.mark.parametrize(
    ("kwargs", "absent"),
    [
        ({"light_verbs": False}, "tud"),
        ({"parliamentary": False}, "tisztelt"),
        ({"spacy": False}, "ahhoz"),
    ],
)
def test_stopword_groups_are_switchable(kwargs, absent):
    assert absent not in hungarian_stopwords(**kwargs)


def test_hungarian_stopwords_accepts_extra():
    assert "sajátszó" in hungarian_stopwords(extra=["sajátszó"])
