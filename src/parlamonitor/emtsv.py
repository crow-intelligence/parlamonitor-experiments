"""Client for the emtsv Hungarian NLP toolchain running as a local REST service.

The service is the ``mtaril/emtsv`` container::

    docker run --rm -d --name emtsv -p 5000:5000 mtaril/emtsv

Its contract is easy to get wrong, so it is worth stating exactly:

* The **module chain is the URL path**, not a JSON field:
  ``POST /tok/morph/pos``. There is no ``lemma`` or ``lem`` module -- asking for
  one raises ``KeyError`` inside ``xtsv`` and the service answers 500. Lemmas
  come out of ``pos``.
* The request body is **multipart form data** with a ``text`` (or ``file``)
  field, not JSON.
* The response is **TSV**, not JSON: a header row naming the columns, one row
  per token, and a blank line between sentences.

The ``anas`` column, which ``morph`` emits, carries every candidate analysis of
every token and is far larger than the rest of the response put together -- the
5,281-word speech in cycle 43 comes back as 2.5 MB of TSV. It is parsed away
here and never returned.

Failures are loud. :func:`analyse` raises :class:`EmtsvError` rather than
returning the text it was given: a caller that silently accepts unanalysed
input produces a corpus that looks complete and is not.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:5000"
"""Where the container publishes the REST API by default."""

DEFAULT_MODULES = "tok/morph/pos"
"""Module chain: tokenise, morphologically analyse, then disambiguate.

``pos`` is what turns the candidate analyses into the single ``lemma`` and
``xpostag`` chosen in context.
"""

CONTENT_CATEGORIES: tuple[str, ...] = ("N", "V", "Adj", "Adv")
"""emMorph main categories kept by :func:`is_content_word` by default."""


class EmtsvError(RuntimeError):
    """The service could not analyse a text, or answered something unusable."""


@dataclass(frozen=True, slots=True)
class Token:
    """One analysed token.

    Attributes:
        form: The surface form, exactly as it appeared in the input.
        lemma: The disambiguated lemma chosen by ``pos``.
        xpostag: The emMorph tag, e.g. ``[/N][Ine]`` or ``[/V][Pst.Def.3Sg]``.
        wsafter: The whitespace that followed the token in the input, decoded
            from the ``wsafter`` column ``tok`` emits. Empty when the chain
            produced no such column. This is what makes input line boundaries
            recoverable from a batched response -- see :func:`analyse_lines`.
    """

    form: str
    lemma: str
    xpostag: str
    wsafter: str = ""


def parse_xpostag(xpostag: str) -> tuple[str | None, tuple[str, ...]]:
    """Split an emMorph tag into its main category and its subcategories.

    Only the first bracket group carries the part of speech; the groups after
    it are inflection (``[Nom]``, ``[Pst.Def.3Sg]``) and are ignored.

    Args:
        xpostag: An emMorph tag as ``pos`` writes it.

    Returns:
        A ``(category, subcategories)`` pair. ``category`` is ``None`` for
        anything without a leading ``[/``, which is how punctuation
        (``[Punct]``) and unanalysed tokens present themselves.

    Example:
        >>> parse_xpostag("[/N][Ine]")
        ('N', ())
        >>> parse_xpostag("[/N|Pro|(Post)][Nom]")
        ('N', ('Pro', '(Post)'))
        >>> parse_xpostag("[Punct]")
        (None, ())
        >>> parse_xpostag("")
        (None, ())
    """
    if not xpostag.startswith("[/"):
        return None, ()
    end = xpostag.find("]")
    if end == -1:
        return None, ()
    parts = xpostag[2:end].split("|")
    return parts[0], tuple(parts[1:])


def is_content_word(
    xpostag: str,
    *,
    keep: tuple[str, ...] = CONTENT_CATEGORIES,
    drop_pronouns: bool = True,
) -> bool:
    """Report whether a tag belongs to a content word.

    Filtering by tag is a stronger lever on topic quality than any frequency
    threshold, because it removes the closed classes outright: determiners,
    conjunctions, postpositions, punctuation, and numerals.

    Args:
        xpostag: An emMorph tag.
        keep: Main categories treated as content. Defaults to nouns, verbs,
            adjectives and adverbs -- ``("N", "V", "Adj", "Adv")``.
        drop_pronouns: Whether to reject tags carrying the ``Pro``
            subcategory. Pronouns are tagged under ``N`` in emMorph, so they
            survive a category-only filter. Defaults to ``True``.

    Returns:
        ``True`` if the token should be kept.

    Example:
        >>> is_content_word("[/N][Ine]")
        True
        >>> is_content_word("[/V][Pst.Def.3Sg]")
        True
        >>> is_content_word("[/Det|Art.Def]")
        False
        >>> is_content_word("[Punct]")
        False

        Pronouns are nouns as far as the category goes, hence the flag:

        >>> is_content_word("[/N|Pro|(Post)][Nom]")
        False
        >>> is_content_word("[/N|Pro|(Post)][Nom]", drop_pronouns=False)
        True
    """
    category, subcategories = parse_xpostag(xpostag)
    if category not in keep:
        return False
    return not (drop_pronouns and "Pro" in subcategories)


def decode_wsafter(field: str) -> str:
    r"""Decode one ``wsafter`` cell into the whitespace it stands for.

    ``tok`` writes the column as a quoted, escaped string literal -- ``" "``
    for a space, ``"\n"`` for a newline, ``""`` for nothing at all -- so it
    survives being carried inside a tab-separated file.

    Args:
        field: The raw cell text.

    Returns:
        The whitespace itself. A cell that is not a parseable string literal
        is returned with its surrounding quotes stripped, which is the best
        available reading and never raises.

    Example:
        >>> decode_wsafter('" "')
        ' '
        >>> decode_wsafter('"\\n"')
        '\n'
        >>> decode_wsafter('""')
        ''
    """
    try:
        decoded = json.loads(field)
    except (ValueError, TypeError):
        return field.strip('"')
    return decoded if isinstance(decoded, str) else field


def parse_tsv_sentences(payload: str) -> list[list[Token]]:
    r"""Parse an emtsv TSV response, preserving its sentence boundaries.

    Columns are located by name from the header row rather than by position,
    so a different module chain that reorders or adds columns still parses.
    Blank lines are emtsv's sentence separators; here they end a sentence
    instead of being skipped.

    Sentences matter for collocation work: a bigram must not be allowed to
    form across a full stop, and one parenthetical line routinely holds
    several sentences joined by ``. `` or `` - ``.

    Args:
        payload: The decoded response body.

    Returns:
        One list of :class:`Token` per sentence, in document order. Empty
        sentences are never emitted.

    Raises:
        EmtsvError: If the payload is empty, or if the header lacks the
            ``form``, ``lemma`` or ``xpostag`` column. A chain of ``tok``
            alone produces no ``lemma``, and that should not pass silently.

    Example:
        >>> payload = (
        ...     "form\twsafter\tlemma\txpostag\n"
        ...     'Taps\t""\ttaps\t[/N][Nom]\n'
        ...     '.\t" "\t.\t[Punct]\n'
        ...     "\n"
        ...     'Derültség\t""\tderültség\t[/N][Nom]\n'
        ... )
        >>> [[t.lemma for t in sentence] for sentence in parse_tsv_sentences(payload)]
        [['taps', '.'], ['derültség']]
    """
    lines = payload.splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.strip()), None)
    if header_index is None:
        raise EmtsvError("emtsv returned an empty response")

    header = lines[header_index].split("\t")
    try:
        columns = {name: header.index(name) for name in ("form", "lemma", "xpostag")}
    except ValueError as exc:
        raise EmtsvError(
            f"emtsv response has no {exc.args[0].split()[0]!s} column; "
            f"header was {header!r}. Does the module chain include `pos`?"
        ) from exc
    # `wsafter` is optional: only chains that start with `tok` emit it.
    ws_index = header.index("wsafter") if "wsafter" in header else None

    width = max(columns.values()) + 1
    sentences: list[list[Token]] = []
    current: list[Token] = []
    for line in lines[header_index + 1 :]:
        if not line.strip():
            if current:
                sentences.append(current)
                current = []
            continue
        fields = line.split("\t")
        if len(fields) < width:
            continue
        wsafter = ""
        if ws_index is not None and len(fields) > ws_index:
            wsafter = decode_wsafter(fields[ws_index])
        current.append(
            Token(
                form=fields[columns["form"]],
                lemma=fields[columns["lemma"]],
                xpostag=fields[columns["xpostag"]],
                wsafter=wsafter,
            )
        )
    if current:
        sentences.append(current)
    return sentences


def parse_tsv(payload: str) -> list[Token]:
    r"""Parse an emtsv TSV response into a flat list of tokens.

    A thin flattening of :func:`parse_tsv_sentences`; use that one when
    sentence boundaries matter.

    Args:
        payload: The decoded response body.

    Returns:
        One :class:`Token` per token row, in document order. Sentence
        boundaries are not preserved.

    Raises:
        EmtsvError: If the payload is empty, or if the header lacks the
            ``form``, ``lemma`` or ``xpostag`` column.

    Example:
        >>> payload = (
        ...     "form\twsafter\tlemma\txpostag\n"
        ...     'Parlamentben\t" "\tparlament\t[/N][Ine]\n'
        ...     "\n"
        ...     'felszólalt\t""\tfelszólal\t[/V][Pst.NDef.3Sg]\n'
        ... )
        >>> [(t.form, t.lemma) for t in parse_tsv(payload)]
        [('Parlamentben', 'parlament'), ('felszólalt', 'felszólal')]
    """
    return [token for sentence in parse_tsv_sentences(payload) for token in sentence]


def analyse(
    text: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    modules: str = DEFAULT_MODULES,
    timeout: float = 300.0,
    retries: int = 3,
    backoff: float = 2.0,
    session: requests.Session | None = None,
) -> list[Token]:
    """Send one text through emtsv and return its analysed tokens.

    Args:
        text: The text to analyse. Must contain something other than
            whitespace.
        base_url: Root URL of the service. Defaults to
            ``http://127.0.0.1:5000``.
        modules: The module chain, as a URL path fragment. Defaults to
            ``tok/morph/pos``.
        timeout: Per-attempt timeout in seconds. Defaults to 300, which is
            generous on purpose: the first request after container start pays
            for loading the morphological analyser and can take ~40 seconds,
            against ~6 seconds warm for a 5,000-word speech.
        retries: Total attempts before giving up. Defaults to 3.
        backoff: Seconds to wait after the first failure, doubling thereafter.
            Defaults to 2.0.
        session: A :class:`requests.Session` to reuse. Passing one matters
            across a corpus -- it keeps the TCP connection open.

    Returns:
        Every token of the text, in order, with ``anas`` discarded.

    Raises:
        ValueError: If ``text`` is empty or whitespace only.
        EmtsvError: If every attempt failed, or the response was unusable.
            Never returns the input text as a fallback: an unanalysed speech
            that looks analysed is worse than a missing one.

    Example:
        >>> tokens = analyse("A kormány benyújtotta a törvényjavaslatot.")
        ... # doctest: +SKIP
        >>> [t.lemma for t in tokens]  # doctest: +SKIP
        ['a', 'kormány', 'benyújt', 'a', 'törvényjavaslat', '.']
    """
    if not text.strip():
        raise ValueError("cannot analyse empty text")
    if retries < 1:
        raise ValueError(f"retries must be at least 1, got {retries}")

    url = f"{base_url.rstrip('/')}/{modules.strip('/')}"
    poster = session or requests
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = poster.post(url, files={"text": (None, text)}, timeout=timeout)
            response.raise_for_status()
            # emtsv sends no charset, and requests would fall back to
            # ISO-8859-1 and mangle every accented Hungarian character.
            return parse_tsv(response.content.decode("utf-8"))
        except (requests.RequestException, EmtsvError, UnicodeDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(backoff * 2 ** (attempt - 1))

    raise EmtsvError(
        f"emtsv failed after {retries} attempt(s) at {url}: {last_error}"
    ) from last_error


def lemmatize(
    text: str,
    *,
    content_only: bool = True,
    keep: tuple[str, ...] = CONTENT_CATEGORIES,
    drop_pronouns: bool = True,
    **kwargs: object,
) -> list[str]:
    """Analyse a text and return its lemmas.

    Case is left exactly as emtsv produced it, which keeps the proper noun
    ``Magyar`` distinct from the adjective ``magyar``. Downstream vectorisers
    generally lowercase anyway; doing it here would throw the distinction away
    before bigram detection could use it.

    Args:
        text: The text to analyse.
        content_only: Whether to keep only content words. Defaults to
            ``True``.
        keep: Main categories treated as content. See :func:`is_content_word`.
        drop_pronouns: Whether to drop pronouns. Defaults to ``True``.
        **kwargs: Passed through to :func:`analyse` (``base_url``, ``timeout``,
            ``retries``, ``session``, ...).

    Returns:
        The lemmas, in document order. Possibly empty, if the text held
        nothing but function words and punctuation.

    Raises:
        ValueError: If ``text`` is empty or whitespace only.
        EmtsvError: If the service could not analyse the text.
    """
    tokens = analyse(text, **kwargs)  # ty: ignore[invalid-argument-type]
    if not content_only:
        return [token.lemma for token in tokens]
    return [
        token.lemma
        for token in tokens
        if is_content_word(token.xpostag, keep=keep, drop_pronouns=drop_pronouns)
    ]


DEFAULT_BATCH_LINES = 250
"""Maximum input lines per batched request in :func:`analyse_lines`."""

DEFAULT_BATCH_WORDS = 3000
"""Maximum whitespace-separated words per batched request.

Both caps apply; whichever is reached first closes the batch. The word cap is
what keeps a single 743-word parenthetical from being batched with 249 others
into a request whose ``anas`` column runs to tens of megabytes.
"""


class AlignmentError(EmtsvError):
    """A batched response could not be mapped back onto its input lines."""


def _chunk_lines(
    lines: Sequence[str], max_lines: int, max_words: int
) -> Iterator[tuple[int, list[str]]]:
    """Group lines into batches under both caps, yielding ``(start, batch)``."""
    if max_lines < 1:
        raise ValueError(f"max_lines must be at least 1, got {max_lines}")
    if max_words < 1:
        raise ValueError(f"max_words must be at least 1, got {max_words}")

    start = 0
    batch: list[str] = []
    words = 0
    for index, line in enumerate(lines):
        n_words = len(line.split())
        # A line longer than the whole budget still goes out on its own rather
        # than being dropped or split mid-sentence.
        if batch and (len(batch) >= max_lines or words + n_words > max_words):
            yield start, batch
            start, batch, words = index, [], 0
        batch.append(line)
        words += n_words
    if batch:
        yield start, batch


def split_by_input_line(
    sentences: Iterable[list[Token]], n_lines: int
) -> list[list[list[Token]]]:
    r"""Regroup a batched response's sentences under the input lines they came from.

    ``tok`` records the whitespace it consumed after every token, so the
    newline that separated two input lines survives in that token's
    :attr:`Token.wsafter`. Counting those newlines is exact, and needs no
    character-offset arithmetic against the request body.

    A sentence is split if a newline falls inside it. That should not happen --
    ``tok`` breaks sentences at newlines -- but silently merging two input
    lines into one sentence would corrupt every downstream collocation.

    Args:
        sentences: Sentences as :func:`parse_tsv_sentences` returns them, for
            a request body of ``n_lines`` newline-terminated lines.
        n_lines: How many input lines the batch held.

    Returns:
        ``n_lines`` entries, each a list of sentences. An input line that
        analysed to nothing contributes an empty list.

    Raises:
        AlignmentError: If the newlines in the response do not account for
            exactly ``n_lines`` input lines.

    Example:
        >>> a = Token("Taps", "taps", "[/N][Nom]")
        >>> b = Token(".", ".", "[Punct]", wsafter="\n")
        >>> c = Token("Csenget", "cseng", "[/V]", wsafter="\n")
        >>> grouped = split_by_input_line([[a, b], [c]], 2)
        >>> [[[t.lemma for t in s] for s in line] for line in grouped]
        [[['taps', '.']], [['cseng']]]
    """
    grouped: list[list[list[Token]]] = []
    current_line: list[list[Token]] = []
    current_sentence: list[Token] = []

    for sentence in sentences:
        for token in sentence:
            current_sentence.append(token)
            for _ in range(token.wsafter.count("\n")):
                if current_sentence:
                    current_line.append(current_sentence)
                    current_sentence = []
                grouped.append(current_line)
                current_line = []
        if current_sentence:
            current_line.append(current_sentence)
            current_sentence = []
    if current_sentence:
        current_line.append(current_sentence)
    if current_line:
        grouped.append(current_line)

    if len(grouped) != n_lines:
        raise AlignmentError(
            f"batched response accounts for {len(grouped)} input line(s), "
            f"expected {n_lines}"
        )
    return grouped


def _post(
    text: str,
    url: str,
    *,
    timeout: float,
    retries: int,
    backoff: float,
    session: requests.Session | None,
) -> str:
    """POST one request body and return the decoded TSV, with retries."""
    poster = session or requests
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = poster.post(url, files={"text": (None, text)}, timeout=timeout)
            response.raise_for_status()
            # emtsv sends no charset, and requests would fall back to
            # ISO-8859-1 and mangle every accented Hungarian character.
            return response.content.decode("utf-8")
        except (requests.RequestException, UnicodeDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(backoff * 2 ** (attempt - 1))
    raise EmtsvError(
        f"emtsv failed after {retries} attempt(s) at {url}: {last_error}"
    ) from last_error


def analyse_lines(
    lines: Sequence[str],
    *,
    base_url: str = DEFAULT_BASE_URL,
    modules: str = DEFAULT_MODULES,
    batch_lines: int = DEFAULT_BATCH_LINES,
    batch_words: int = DEFAULT_BATCH_WORDS,
    timeout: float = 600.0,
    retries: int = 3,
    backoff: float = 2.0,
    session: requests.Session | None = None,
    on_batch: Callable[[int, int], None] | None = None,
) -> list[list[list[Token]]]:
    r"""Analyse many short texts, batched, and return each one's sentences.

    One request per batch rather than per line. On a corpus of 64,052 distinct
    parenthetical lines that is the difference between a few minutes and about
    an hour, because the per-request cost is dominated by HTTP and by emtsv
    re-entering its module chain, not by the handful of words in a line.

    Batching is only safe if the response can be mapped back onto the input
    with certainty. Two independent checks enforce that, per batch:

    1. The tokens must reconstruct the request body exactly, from ``form`` and
       ``wsafter``.
    2. The newlines in ``wsafter`` must account for exactly as many input
       lines as were sent.

    A batch failing either check is **re-sent one line at a time**, so a
    tokenizer quirk costs speed rather than correctness. A line that still
    fails alone raises.

    Args:
        lines: The texts, one per entry. Empty or whitespace-only entries are
            allowed and analyse to no sentences at all.
        base_url: Root URL of the service. Defaults to
            ``http://127.0.0.1:5000``.
        modules: The module chain. Defaults to ``tok/morph/pos``. Must include
            ``tok``, since ``wsafter`` is what carries the line boundaries.
        batch_lines: Maximum lines per request. Defaults to 250.
        batch_words: Maximum words per request. Defaults to 3000.
        timeout: Per-attempt timeout in seconds. Defaults to 600.
        retries: Attempts per request before giving up. Defaults to 3.
        backoff: Seconds to wait after the first failure, doubling. Defaults
            to 2.0.
        session: A :class:`requests.Session` to reuse. Worth passing -- it
            keeps the TCP connection open across hundreds of batches.
        on_batch: Optional callable invoked as ``on_batch(n_done, n_total)``
            after each batch, for progress reporting.

    Returns:
        One entry per input line, each a list of sentences, each a list of
        :class:`Token`.

    Raises:
        EmtsvError: If a single line could not be analysed even on its own, or
            the service never answered.
        ValueError: If the batch caps are below 1.

    Example:
        >>> analyse_lines(["Taps.", "Az elnök csenget."])  # doctest: +SKIP
        [[[Token(form='Taps', ...)]], [[Token(form='Az', ...)]]]
    """
    url = f"{base_url.rstrip('/')}/{modules.strip('/')}"
    results: list[list[list[Token]]] = []
    done = 0

    for _, batch in _chunk_lines(lines, batch_lines, batch_words):
        # Every line is newline-terminated, so the final line's boundary is
        # recorded in `wsafter` exactly like the others.
        body = "".join(line + "\n" for line in batch)
        if not body.strip():
            results.extend([[] for _ in batch])
        else:
            payload = _post(
                body,
                url,
                timeout=timeout,
                retries=retries,
                backoff=backoff,
                session=session,
            )
            try:
                sentences = parse_tsv_sentences(payload)
                _check_reconstruction(sentences, body)
                results.extend(split_by_input_line(sentences, len(batch)))
            except (AlignmentError, EmtsvError):
                results.extend(
                    _analyse_one_by_one(
                        batch,
                        url,
                        timeout=timeout,
                        retries=retries,
                        backoff=backoff,
                        session=session,
                    )
                )
        done += len(batch)
        if on_batch is not None:
            on_batch(done, len(lines))

    return results


def _check_reconstruction(sentences: Iterable[list[Token]], body: str) -> None:
    """Raise unless the tokens rebuild the request body character for character."""
    rebuilt = "".join(
        token.form + token.wsafter for sentence in sentences for token in sentence
    )
    if rebuilt != body:
        raise AlignmentError(
            f"tokens rebuild {len(rebuilt)} characters, request body was "
            f"{len(body)}; the response cannot be aligned to its input"
        )


def _analyse_one_by_one(
    batch: Sequence[str],
    url: str,
    *,
    timeout: float,
    retries: int,
    backoff: float,
    session: requests.Session | None,
) -> list[list[list[Token]]]:
    """Fall back to one request per line when a batch would not align."""
    out: list[list[list[Token]]] = []
    for line in batch:
        if not line.strip():
            out.append([])
            continue
        payload = _post(
            line + "\n",
            url,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
            session=session,
        )
        out.append(parse_tsv_sentences(payload))
    return out
