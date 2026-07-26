# Reflection

With more data, I'd add 150-200+ tickets per category instead of ~10 — with only 40
total examples the TF-IDF vocabulary is sparse, so predicted probabilities stay spread
across classes and most tickets trip the 60% review threshold even when the top label
is correct. More data would sharpen confidence, not just accuracy. I'd also add a true
"none of the above" class instead of forcing every ticket into one of four buckets,
since real queues always get spam/unrelated messages. Word embeddings (or a small
fine-tuned transformer) would help more than bigrams once volume justifies the extra
latency. Finally, I'd log every low-confidence ticket + human-assigned label and
periodically retrain on it, so the model improves from its own mistakes over time.
