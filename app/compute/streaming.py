"""
Decides when text the Composer is still writing may be shown to a reader.

The problem this exists for: everywhere else, a product streams its model's
output straight to the screen because nothing sits between the two. Here
something does. verify_numbers checks the finished draft against the facts
payload, and a draft quoting a figure the payload does not contain is discarded
in favour of a template. Streaming naively would mean showing numbers and then
withdrawing them, in a product whose entire design is that it never states a
figure it cannot source.

So text is released in COMPLETE UNITS — a paragraph, a bullet line, a sentence —
and only once every number inside that unit traces back to the payload. Two
consequences, and both are the reason for the choice:

  Nothing is ever retracted. An invented figure is caught while its line is
  still in the buffer, so it is never displayed at all. The stream stops and the
  template takes over from text the reader has already seen as settled.

  The markdown is never broken. A complete line has balanced "**", so an
  ordinary markdown renderer works on each released chunk. Releasing on token
  boundaries would emit "**Real GD" and leave the frontend to cope.

What it costs, stated plainly: the reveal is line by line rather than word by
word. Smoother than one block of text landing at once, chunkier than a product
with no verifier. That is the price of the guarantee, and it is not a limitation
to be engineered away later — it is the guarantee.

The gate is deliberately not the last word. verify_numbers still runs on the
complete text afterwards, because this checks numbers in isolation while the
final check also sees the answer as a whole.
"""
import re
from typing import Iterator, Optional

from collections import Counter

from app.compute.verifier import (NUMBER_PATTERN, MAX_LINE_REPEATS, allowed_numbers,
                                   is_allowed_number, looks_degenerate)

# Where a unit may end. A blank line closes a paragraph; a newline closes a
# bullet; a sentence-ending punctuation mark FOLLOWED BY WHITESPACE closes a
# sentence. Arabic full stops and question marks are here for the same reason
# the rest of this codebase carries both languages.
#
# There is deliberately no end-of-buffer alternative. An earlier version had
# `(?<=[.!?])$`, on the reasoning that a trailing full stop ends a sentence —
# but during streaming the end of the buffer is not the end of anything, it is
# just where the model has got to. "…measured from 166." matched it, the gate
# released that as a finished sentence, and checked "166." as a number: 166.11
# is in the payload and 166 is not, so a correct answer was rejected and
# replaced by the template mid-stream. The whitespace requirement is what
# distinguishes a full stop from a decimal point, and it has to be present.
# The tail is flushed by finish() instead, once the model has actually stopped.
_BOUNDARY = re.compile(r"(\n\n|\n|(?<=[.!?؟।])\s+)")

# Below this, holding back for a boundary costs more in latency than the
# smoothness is worth — a long paragraph with no punctuation should still reach
# the reader in pieces. Releasing at a space is safe for markdown as long as no
# emphasis marker is open, which _has_open_emphasis checks.
SOFT_RELEASE_CHARS = 220


def _has_open_emphasis(text: str) -> bool:
    """Whether releasing here would cut a markdown emphasis run in half."""
    return text.count("**") % 2 == 1 or (text.count("*") - 2 * text.count("**")) % 2 == 1


def _trailing_partial_number(text: str) -> bool:
    """Whether the text ends mid-number, with nothing after it yet.

    "185.1" may still become "185.17", and checking it as written would reject a
    figure the model had not finished writing.

    \\Z, not $. In Python $ also matches just before a trailing newline, so a
    finished bullet line like "- Global Rank: 11\\n" would read as ending
    mid-number and never be released — the stream would stall on any line whose
    last word is a figure. A number followed by ANY character, whitespace
    included, is complete.
    """
    return bool(re.search(r"\d[\d,]*\.?\d*\Z", text))


class AnswerGate:
    """Buffers model output and releases what is safe to display.

    Usage:
        gate = AnswerGate(payload)
        for piece in chat_stream(...):
            for chunk in gate.push(piece):
                emit(chunk)
        for chunk in gate.finish():
            emit(chunk)
        if gate.rejected: ...
    """

    def __init__(self, facts_payload: dict):
        self._allowed = allowed_numbers(facts_payload)
        self._buffer = ""
        self._seen = Counter()
        self.released = ""
        # Why the stream was abandoned, if it was. Whatever has been released is
        # valid; the caller replaces the rest.
        self.rejected: Optional[str] = None
        # WHICH kind of failure: "number" for a figure that is not in the
        # payload, "degenerate" for the model coming off the rails. The caller
        # answers them differently — a malformed draft is worth asking for
        # again, an invented figure is not — and it cannot tell them apart from
        # the text, because the offending unit was never released.
        self.rejected_kind: Optional[str] = None

    def _numbers_clear(self, unit: str) -> Optional[str]:
        """The first number in `unit` that is not in the payload, or None."""
        for match in NUMBER_PATTERN.findall(unit):
            if not is_allowed_number(match, self._allowed):
                return match
        return None

    def _sane(self, unit: str) -> Optional[str]:
        """Whether this unit shows the model has come off the rails.

        Checked per unit and not only at the end, because the end is too late
        when the text is already on screen. A composed answer once restated
        itself eight times and drifted into Chinese; streamed, the reader would
        have watched it happen.

        Repetition is counted across the whole answer rather than within one
        unit, which is the only way to see a loop — each pass through it looks
        perfectly reasonable on its own.
        """
        broken = looks_degenerate(unit)
        if broken:
            return broken
        key = " ".join(unit.split())
        if len(key) > 20:
            self._seen[key] += 1
            if self._seen[key] > MAX_LINE_REPEATS:
                return f"answer repeats the same line {self._seen[key]} times"
        return None

    def push(self, piece: str) -> Iterator[str]:
        """Takes a raw fragment; yields whatever is now safe to show."""
        if self.rejected:
            return
        self._buffer += piece
        while True:
            unit, rest = self._split()
            if unit is None:
                return
            bad_number = self._numbers_clear(unit)
            offender = bad_number or self._sane(unit)
            if offender is not None:
                # Caught before display. The buffer is dropped rather than
                # released — this is the whole point of holding it.
                self.rejected = offender
                self.rejected_kind = "number" if bad_number else "degenerate"
                self._buffer = ""
                return
            self._buffer = rest
            self.released += unit
            yield unit

    def _split(self) -> tuple[Optional[str], str]:
        """The next releasable unit and what remains, or (None, buffer)."""
        match = None
        for candidate in _BOUNDARY.finditer(self._buffer):
            match = candidate
        if match:
            end = match.end()
            unit, rest = self._buffer[:end], self._buffer[end:]
            # A boundary inside an unclosed "**" is not a boundary — the
            # emphasis would render broken. Wait for more text instead.
            #
            # Nor is one that leaves a half-written number at the end: checking
            # "166." as though it were final rejects an answer whose next
            # character was going to be a "1". The boundary pattern no longer
            # produces that case on its own, and this stays as the guarantee
            # rather than as a second line of defence — no release path may
            # ever hand a partial number to the verifier.
            if not _has_open_emphasis(unit) and not _trailing_partial_number(unit):
                return unit, rest
        # No boundary, but the buffer has grown long enough that waiting is
        # worse than releasing at the last space.
        if len(self._buffer) >= SOFT_RELEASE_CHARS:
            cut = self._buffer.rfind(" ")
            if cut > 0:
                unit, rest = self._buffer[:cut + 1], self._buffer[cut + 1:]
                if not _has_open_emphasis(unit) and not _trailing_partial_number(unit):
                    return unit, rest
        return None, self._buffer

    def finish(self) -> Iterator[str]:
        """Releases the tail once the model has stopped.

        The last sentence usually has no trailing boundary, so without this the
        final line of every answer would be held back forever.
        """
        if self.rejected or not self._buffer:
            return
        tail, self._buffer = self._buffer, ""
        bad_number = self._numbers_clear(tail)
        offender = bad_number or self._sane(tail)
        if offender is not None:
            self.rejected = offender
            self.rejected_kind = "number" if bad_number else "degenerate"
            return
        self.released += tail
        yield tail
