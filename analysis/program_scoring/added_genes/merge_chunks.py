"""Reassemble chunked score_variants_python.py outputs <stem>__c<i>of<n>.npz into <stem>.npz in the variants.json order.
  python merge_chunks.py <variants.json> <stem> <n>"""
import json, os, sys
import numpy as np
vfile, stem, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
vids = list(json.load(open(vfile)))
parts = [np.load(f"{stem}__c{i}of{n}.npz", allow_pickle=True) for i in range(n)]
cand = list(parts[0]["cand"]); assert all(list(p["cand"]) == cand for p in parts)
got = {v: p["S"][j] for p in parts for j, v in enumerate(p["variants"])}
assert sorted(got) == sorted(vids), (len(got), len(vids))
np.savez_compressed(f"{stem}.npz", S=np.stack([got[v] for v in vids]), variants=np.array(vids), cand=np.array(cand))
for i in range(n):
    os.remove(f"{stem}__c{i}of{n}.npz")
print("merged", stem, len(vids))
