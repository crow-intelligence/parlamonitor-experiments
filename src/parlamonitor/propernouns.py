"""Repair of proper nouns that emMorph analyses as common words.

emtsv's morphological analyser has no gazetteer of Hungarian party names, so
it decomposes the ones that happen to be ordinary Hungarian words. The
consequential case in the parentheticals corpus is **Jobbik**, which the
analyser reads as the comparative adjective *jobbik* ("the better one") and
lemmatises to ``jó``::

    Jobbik -> jó   [/Adj][_Comp/Adj][_Design/Adj][Nom]

That is not a tuning choice to be swept; it is wrong, and it is wrong 20,705
times -- ``Jobbik`` is among the most frequent content words in the corpus.
Left alone it both erases the party and inflates ``jó``, and it puts a
nonsense token into every collocation the party takes part in.

The repair is deliberately narrow and auditable:

* it is keyed on the **surface form**, not on the lemma, so it cannot fire on
  a token that merely happens to share a lemma;
* it only fires on a **capitalised** form, which is what distinguishes the
  party ``Jobbik`` (20,705 occurrences) from the adjective ``jobbik`` (5);
* every entry is listed in :data:`PROPER_NOUN_LEMMAS` and the number of tokens
  it touched is reported in the run manifest.

Forms that emMorph already gets right -- ``Fidesz``, ``KDNP``, ``MSZP``,
``LMP``, ``DK``, ``SZDSZ``, ``MDF`` all lemmatise to themselves -- are not
listed here. Adding them would change nothing and would make the list look
like it is doing more work than it is.

**Known limitation, not repaired.** The multi-word name *Mi Hazánk* is
analysed as the pronoun ``mi`` plus ``haza[Poss.1Pl]``, and repairing it
would need a rule spanning two tokens. It is left alone; the collocation pass
recovers it as the fused bigram ``mi#haza``, which is interpretable, and this
is recorded rather than hidden.
"""

from __future__ import annotations

from collections.abc import Iterable

from parlamonitor.emtsv import Token

PROPER_NOUN_LEMMAS: dict[str, tuple[str, str]] = {
    # form: (lemma, xpostag) -- party names emMorph decomposes into common words.
    "Jobbik": ("Jobbik", "[/N][Nom]"),
    "Momentum": ("Momentum", "[/N][Nom]"),
    "Párbeszéd": ("Párbeszéd", "[/N][Nom]"),
    "Együtt": ("Együtt", "[/N][Nom]"),
}
"""Surface form to the ``(lemma, xpostag)`` it should have had.

Verified against the corpus and against emtsv, entry by entry. ``Jobbik``
(20,705 occurrences) becomes ``jó``; ``Együtt`` (7) becomes the adverb
``együtt``. ``Momentum`` (637) and ``Párbeszéd`` (1,409) are lemmatised
correctly in substance but lowercased, which merges each party with the common
noun -- *párbeszéd* ("dialogue") is a word this corpus also uses in its
ordinary sense. Pinning the capitalised form keeps the two apart.

Nothing is listed that does not occur: ``Lendülettel`` would be a plausible
entry and appears zero times, so it is not here.

The tag is set to a plain nominative noun. Case marking on a party name is not
information any count here uses, and inventing a plausible-looking inflection
tag would be inventing data.
"""


def repair_token(token: Token) -> tuple[Token, bool]:
    """Apply the proper-noun override to one token.

    Args:
        token: A token as emtsv analysed it.

    Returns:
        A ``(token, repaired)`` pair. ``repaired`` is ``True`` only when the
        override actually fired, so callers can count how often it did.

    Example:
        >>> broken = Token("Jobbik", "jó", "[/Adj][_Comp/Adj][_Design/Adj][Nom]")
        >>> fixed, repaired = repair_token(broken)
        >>> fixed.lemma, fixed.xpostag, repaired
        ('Jobbik', '[/N][Nom]', True)

        The lowercase adjective is left exactly as emtsv read it:

        >>> repair_token(Token("jobbik", "jó", "[/Adj][_Comp/Adj]"))[1]
        False

        And so is anything not on the list:

        >>> repair_token(Token("Fidesz", "Fidesz", "[/N][Nom]"))[1]
        False
    """
    replacement = PROPER_NOUN_LEMMAS.get(token.form)
    if replacement is None:
        return token, False
    lemma, xpostag = replacement
    if token.lemma == lemma and token.xpostag == xpostag:
        return token, False
    return Token(
        form=token.form, lemma=lemma, xpostag=xpostag, wsafter=token.wsafter
    ), True


def repair_sentences(
    sentences: Iterable[list[Token]],
) -> tuple[list[list[Token]], int]:
    """Apply :func:`repair_token` across a document's sentences.

    Args:
        sentences: Sentences of tokens.

    Returns:
        A ``(sentences, n_repaired)`` pair.

    Example:
        >>> sents = [[Token("Taps", "taps", "[/N][Nom]"),
        ...           Token("Jobbik", "jó", "[/Adj][_Comp/Adj]")]]
        >>> repaired, n = repair_sentences(sents)
        >>> [t.lemma for t in repaired[0]], n
        (['taps', 'Jobbik'], 1)
    """
    out: list[list[Token]] = []
    n_repaired = 0
    for sentence in sentences:
        repaired_sentence = []
        for token in sentence:
            token, was_repaired = repair_token(token)
            n_repaired += was_repaired
            repaired_sentence.append(token)
        out.append(repaired_sentence)
    return out, n_repaired
