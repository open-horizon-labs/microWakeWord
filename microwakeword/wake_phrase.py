"""Declarative wake-phrase contracts shared by training and qualification.

The acoustic topology must be derived from the phrase being trained.  Keeping
that identity explicit prevents a replacement phrase from silently inheriting
Hi-Fi Kizz's seven-phone/23-output tensor contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WakePhraseSpec:
    phrase_id: str
    text: str
    ctc_transcript: str
    phones: tuple[str, ...]
    collision_transcripts: tuple[str, ...]
    collision_phones: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not self.phrase_id or not self.text:
            raise ValueError("wake phrase ID and text must not be empty")
        if not self.ctc_transcript or len(self.ctc_transcript) != len(self.phones):
            raise ValueError(
                "the MMS CTC transcript must contain one token per declared phone"
            )
        if not self.collision_transcripts:
            raise ValueError("wake phrase needs explicit collision transcripts")
        if len(self.collision_transcripts) != len(self.collision_phones):
            raise ValueError(
                "every collision transcript needs one declared IPA phone path"
            )
        if any(not path for path in self.collision_phones):
            raise ValueError("collision phone paths must not be empty")
        if self.ctc_transcript in self.collision_transcripts:
            raise ValueError("canonical transcript cannot also be a collision")


HI_FI_KIZZ = WakePhraseSpec(
    phrase_id="hiphi-kizz",
    text="Hi-Fi Kizz",
    ctc_transcript="hifikiz",
    phones=("h", "aɪ", "f", "aɪ", "k", "ɪ", "z"),
    collision_transcripts=(
        "hifikids",
        "hifikiss",
        "highfivekiz",
        "hiffykiz",
        "hippykiz",
    ),
    collision_phones=(
        ("h", "aɪ", "f", "aɪ", "k", "ɪ", "d", "z"),
        ("h", "aɪ", "f", "aɪ", "k", "ɪ", "s"),
        ("h", "aɪ", "f", "aɪ", "v", "k", "ɪ", "z"),
        ("h", "ɪ", "f", "i", "k", "ɪ", "z"),
        ("h", "ɪ", "p", "i", "k", "ɪ", "z"),
    ),
)


KIZZ_CONTROL = WakePhraseSpec(
    phrase_id="kizz-control",
    text="Kizz Control",
    ctc_transcript="kizkontrol",
    phones=("k", "ɪ", "z", "k", "ə", "n", "t", "ɹ", "oʊ", "l"),
    collision_transcripts=(
        "kidskontrol",
        "kiskontrol",
        "thiskontrol",
        "hiskontrol",
        "kizkontroller",
        "kizkontrold",
        "kizpatrol",
        "kitchenkontrol",
        "kidskantrol",
    ),
    collision_phones=(
        ("k", "ɪ", "d", "z", "k", "ə", "n", "t", "ɹ", "oʊ", "l"),
        ("k", "ɪ", "s", "k", "ə", "n", "t", "ɹ", "oʊ", "l"),
        ("ð", "ɪ", "s", "k", "ə", "n", "t", "ɹ", "oʊ", "l"),
        ("h", "ɪ", "z", "k", "ə", "n", "t", "ɹ", "oʊ", "l"),
        ("k", "ɪ", "z", "k", "ə", "n", "t", "ɹ", "oʊ", "l", "ɚ"),
        ("k", "ɪ", "z", "k", "ə", "n", "t", "ɹ", "oʊ", "l", "d"),
        ("k", "ɪ", "z", "p", "ɐ", "t", "ɹ", "oʊ", "l"),
        ("k", "ɪ", "tʃ", "ə", "n", "k", "ə", "n", "t", "ɹ", "oʊ", "l"),
        ("k", "ɪ", "d", "z", "k", "æ", "n", "t", "ɹ", "oʊ", "l"),
    ),
)


WAKE_PHRASES = {
    spec.phrase_id: spec for spec in (HI_FI_KIZZ, KIZZ_CONTROL)
}


def get_wake_phrase(phrase_id: str) -> WakePhraseSpec:
    try:
        return WAKE_PHRASES[phrase_id]
    except KeyError as error:
        raise ValueError(
            f"unknown wake phrase {phrase_id!r}; choose one of {sorted(WAKE_PHRASES)}"
        ) from error


__all__ = [
    "HI_FI_KIZZ",
    "KIZZ_CONTROL",
    "WAKE_PHRASES",
    "WakePhraseSpec",
    "get_wake_phrase",
]
