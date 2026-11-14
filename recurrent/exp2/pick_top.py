import json
import sys

table = json.load(open(sys.argv[1]))["table"]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
top = sorted(table.items(), key=lambda kv: -kv[1]["score"])[:k]
print(",".join(c for c, _ in top))
