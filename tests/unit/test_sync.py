from memex.models import MemoryRecord, datetime_from_timestamp
from memex.sync import CRDTState


def _record(memory_id: str, accessed_at: int, text: str = "memory") -> MemoryRecord:
    timestamp = datetime_from_timestamp(accessed_at)
    return MemoryRecord(
        id=memory_id,
        namespace="default",
        text=text,
        created_at=timestamp,
        accessed_at=timestamp,
    )


def test_crdt_merge_keeps_newest_record() -> None:
    left = CRDTState(records={"one": _record("one", 10, "old")})
    right = CRDTState(records={"one": _record("one", 20, "new")})

    merged = left.merge(right)

    assert merged.records["one"].text == "new"


def test_crdt_tombstone_wins_over_older_record() -> None:
    left = CRDTState(records={"one": _record("one", 10)})
    right = CRDTState(deleted={"one": 20})

    merged = left.merge(right)

    assert "one" not in merged.records
    assert merged.deleted["one"] == 20
