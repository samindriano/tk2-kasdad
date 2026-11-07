# Exact encoding of unigram_order_sha256

Additive clarification; ROOT_VOCAB_PARITY.json and fitted states are unchanged.

The digest binds ordered **(term, original TF-IDF column index)** pairs, not only terms and not a NumPy array digest. Original indices contain gaps after bigrams are excluded; do not replace them with compact embedding-row indices before hashing.

```python
u = sorted(
    [(str(t), int(i)) for t, i in vectorizer.vocabulary_.items() if " " not in t],
    key=lambda pair: pair[1],
)
payload = json.dumps(u, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
digest = hashlib.sha256(payload).hexdigest()
```

Python tuples serialize as JSON arrays. There is no shape/dtype header or newline. This equals the original positional pilot's canonical_sha256(u).

Root independently recomputed both on 7 November 2026 WIB from the saved A02 prefix vectorizers: E67 (5675 unigrams) `c95d01f6d23056101bab4d741c70b81addd6a98591542b6dbee01dc999187c2c`; E75 (5723) `9b6cebb1f3e4c36bdb88d10e17de42c0d8d6ac8cbda019d3f0edddde3229b907`. Both match ROOT_VOCAB_PARITY.json exactly. No query targets, new fits, or model/recipe changes were involved. Compact embedding row order can be derived and hashed separately.
