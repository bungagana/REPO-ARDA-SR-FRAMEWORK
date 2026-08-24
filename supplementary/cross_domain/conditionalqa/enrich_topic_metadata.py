
import json
import sys
from collections import Counter
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CROSS_DOMAIN_DIR = THIS_DIR.parent
sys.path.insert(0, str(CROSS_DOMAIN_DIR))

from generic_metadata import bucket_topic  # noqa: E402

CHUNKS_PATH = THIS_DIR / "kb_conditionalqa" / "chunks.json"


def main():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    counts = Counter()
    for c in chunks:
        topic = bucket_topic(c["filename"])
        if topic:
            c["topic"] = topic
        counts[topic or "(none)"] += 1

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("Topic bucket distribution (by chunk count):")
    for topic, n in counts.most_common():
        print(f"  {topic:20s} {n}")
    print(f"\nSaved enriched chunks -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
