"""Scoring for ZentroMart practice questions (same scorer for every step)."""

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def norm(text):
    """Lowercase, Hindi digits -> 0-9, collapse spaces."""
    return " ".join(str(text).translate(DEV_DIGITS).lower().split())


def split_list(cell):
    """Dataset lists use ' ; ' between items."""
    return [x.strip() for x in str(cell).split(" ; ") if x.strip()]


def decision_ok(pred, acceptable):
    return norm(pred) in [norm(x) for x in split_list(acceptable)]


def include_score(reply, must_include):
    """Fraction of must_include groups found. ' / ' = any one alternative is enough."""
    groups = split_list(must_include)
    if not groups:
        return 1.0
    r = norm(reply)
    hits = sum(any(norm(alt) in r for alt in g.split(" / ")) for g in groups)
    return hits / len(groups)


def violates(reply, must_not_include):
    r = norm(reply)
    return any(norm(x) in r for x in split_list(must_not_include))


def norm_id(x):
    return str(x).strip().strip("[]").upper()


def evidence_recall(pred_ids, gold):
    """Share of gold evidence IDs the agent cited. None if no gold evidence."""
    gold_ids = {norm_id(g) for g in split_list(gold)}
    if not gold_ids:
        return None
    pred = {norm_id(p) for p in pred_ids}
    return len(gold_ids & pred) / len(gold_ids)