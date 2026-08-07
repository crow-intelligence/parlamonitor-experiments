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

import time
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
    """

    form: str
    lemma: str
    xpostag: str


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


def parse_tsv(payload: str) -> list[Token]:
    r"""Parse an emtsv TSV response into tokens.

    Columns are located by name from the header row rather than by position,
    so a different module chain that reorders or adds columns still parses.
    Blank lines are emtsv's sentence separators and carry no token.

    Args:
        payload: The decoded response body.

    Returns:
        One :class:`Token` per token row, in document order. Sentence
        boundaries are not preserved -- ask the ``tok`` output for those.

    Raises:
        EmtsvError: If the payload is empty, or if the header lacks the
            ``form``, ``lemma`` or ``xpostag`` column. A chain of ``tok``
            alone produces no ``lemma``, and that should not pass silently.

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

    width = max(columns.values()) + 1
    tokens: list[Token] = []
    for line in lines[header_index + 1 :]:
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < width:
            continue
        tokens.append(
            Token(
                form=fields[columns["form"]],
                lemma=fields[columns["lemma"]],
                xpostag=fields[columns["xpostag"]],
            )
        )
    return tokens


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
