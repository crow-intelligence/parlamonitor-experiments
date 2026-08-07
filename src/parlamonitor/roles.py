"""Discourse roles: which speeches are question time and which are debate.

The Q&A export is **not** a separate corpus. 622 of the 659 turn uids in
``cycle43-qa.jsonl`` are also among the 1,693 speeches in
``cycle43-speeches.jsonl`` -- the Q&A file reconstructs exchanges out of
speeches the other file already contains. Concatenating the two would count
those 622 twice.

So comparing question time against debate means **partitioning** the speeches,
which is what this module does. The partition can be derived two ways -- from
each speech's own ``speech_type``, or from membership in the Q&A file's turn
list -- and :func:`compare_role_sources` checks that the two agree rather than
quietly trusting one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

DEBATE = "debate"
"""Role assigned to anything that is not part of a question-time exchange."""

ROLE_BY_SPEECH_TYPE: Mapping[str, str] = {
    "elhangzik az interpelláció/kérdés/azonnali kérdés": "question",
    "kérdés megválaszolva": "answer",
    "interpelláció szóban megválaszolva": "answer",
    "azonnali kérdésre adott képviselői viszonválasz": "mp_rejoinder",
    "azonnali kérdésre adott miniszteri viszonválasz": "minister_rejoinder",
    "képviselő elfogadta a választ": "reaction",
    "képviselő elutasította a választ": "reaction",
}
"""Upstream *felszólalás típusa* to discourse role.

Counts in cycle 43: question 203, answer 202 (139 + 63), mp_rejoinder 76,
minister_rejoinder 75, reaction 64 (46 + 18), leaving 1,071 as
:data:`DEBATE`.
"""

QA_ROLES: frozenset[str] = frozenset(
    {"question", "answer", "mp_rejoinder", "minister_rejoinder", "reaction"}
)
"""The roles that make up question time, i.e. everything except debate."""


def discourse_role(speech: Mapping[str, Any]) -> str:
    """Return the discourse role of one speech.

    Args:
        speech: A speech record from
            :func:`parlamonitor.loading.load_speeches`.

    Returns:
        One of ``question``, ``answer``, ``mp_rejoinder``,
        ``minister_rejoinder``, ``reaction``, or :data:`DEBATE`. An
        unrecognised ``speech_type`` falls back to :data:`DEBATE` rather than
        raising, because the type vocabulary is upstream's and grows.

    Example:
        >>> discourse_role({"speech_type": "kérdés megválaszolva"})
        'answer'
        >>> discourse_role({"speech_type": "vezérszónoki felszólalás"})
        'debate'
        >>> discourse_role({})
        'debate'
    """
    return ROLE_BY_SPEECH_TYPE.get(speech.get("speech_type", ""), DEBATE)


def is_qa(speech: Mapping[str, Any]) -> bool:
    """Report whether a speech belongs to a question-time exchange.

    Example:
        >>> is_qa({"speech_type": "kérdés megválaszolva"})
        True
        >>> is_qa({"speech_type": "felszólalás"})
        False
    """
    return discourse_role(speech) in QA_ROLES


def qa_turn_uids(qa_records: Iterable[Mapping[str, Any]]) -> set[str]:
    """Collect every speech uid referenced by the Q&A export's turns.

    Args:
        qa_records: Records from :func:`parlamonitor.loading.load_qa`.

    Returns:
        The uids, including turns whose speech is absent from the speeches
        export (37 of 659 in cycle 43, mostly the unpublished sitting 43020).

    Example:
        >>> qa_turn_uids([{"turns": [{"uid": "43003-1"}, {"uid": None}]}])
        {'43003-1'}
    """
    uids: set[str] = set()
    for record in qa_records:
        for turn in record.get("turns") or ():
            uid = turn.get("uid")
            if uid:
                uids.add(uid)
    return uids


def compare_role_sources(
    speeches: Iterable[Mapping[str, Any]], qa_uids: set[str]
) -> dict[str, Any]:
    """Cross-check the ``speech_type`` partition against the Q&A turn list.

    Two independent derivations of the same partition should agree. Where they
    do not, the disagreement is reported rather than resolved -- a speech typed
    as question time but missing from the Q&A file, or the reverse, is a fact
    about the data worth seeing.

    Args:
        speeches: Speech records.
        qa_uids: Output of :func:`qa_turn_uids`.

    Returns:
        A dict with ``n_speeches``, ``by_role`` (counts), ``agree``,
        ``typed_qa_not_in_qa_file`` and ``in_qa_file_not_typed_qa`` (uid lists,
        capped at 20 each for legibility, with the full counts alongside).

    Example:
        >>> speeches = [
        ...     {"uid": "a", "speech_type": "kérdés megválaszolva"},
        ...     {"uid": "b", "speech_type": "felszólalás"},
        ... ]
        >>> result = compare_role_sources(speeches, {"a"})
        >>> result["agree"], result["by_role"]["answer"]
        (True, 1)
    """
    speeches = list(speeches)
    by_role: dict[str, int] = {}
    typed_qa: set[str] = set()
    all_uids: set[str] = set()

    for speech in speeches:
        role = discourse_role(speech)
        by_role[role] = by_role.get(role, 0) + 1
        uid = speech.get("uid")
        if uid:
            all_uids.add(uid)
            if role in QA_ROLES:
                typed_qa.add(uid)

    # Only uids present in the speeches export can disagree; the Q&A file also
    # references turns whose speech was never published.
    in_file = qa_uids & all_uids
    only_typed = sorted(typed_qa - in_file)
    only_file = sorted(in_file - typed_qa)

    return {
        "n_speeches": len(speeches),
        "by_role": by_role,
        "agree": not only_typed and not only_file,
        "n_typed_qa": len(typed_qa),
        "n_qa_uids_in_speeches": len(in_file),
        "n_typed_qa_not_in_qa_file": len(only_typed),
        "n_in_qa_file_not_typed_qa": len(only_file),
        "typed_qa_not_in_qa_file": only_typed[:20],
        "in_qa_file_not_typed_qa": only_file[:20],
    }
