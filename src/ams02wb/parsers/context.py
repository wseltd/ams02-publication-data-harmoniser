"""Parse context carrying provenance flags for uncertainty labelling."""

from __future__ import annotations

from pydantic import BaseModel


class ParseContext(BaseModel):
    """Flags indicating how each uncertainty was obtained during parsing.

    These booleans drive the uncertainty labeller: the labeller inspects
    these flags to decide whether each uncertainty is PUBLISHED, DERIVED,
    or ASSUMED.  Flags default to False (meaning the value was not found
    in the source and will be labelled ASSUMED).
    """

    stat_err_from_table: bool = False
    sys_err_from_table: bool = False
    sys_err_symmetrised: bool = False
    err_split_heuristic: bool = False

    def __repr__(self) -> str:
        flags = []
        if self.stat_err_from_table:
            flags.append("stat_from_table")
        if self.sys_err_from_table:
            flags.append("sys_from_table")
        if self.sys_err_symmetrised:
            flags.append("sys_symmetrised")
        if self.err_split_heuristic:
            flags.append("err_split")
        return f"ParseContext({', '.join(flags) if flags else 'no flags'})"
