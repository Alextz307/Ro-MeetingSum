import matplotlib.pyplot as plt
import numpy as np

methods = [
    "Baseline (mT5 Abstractive Chunking)",
    "Extractive (MatchSum)",
    "Hybrid (MatchSum + mT5 Abstractive Chunking)",
]
r1_scores = [0.3051, 0.3380, 0.3439]
rl_scores = [0.1837, 0.2523, 0.2367]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(
    x - width / 2, r1_scores, width, label="ROUGE-1 (Content)", color="#4e79a7"
)
rects2 = ax.bar(
    x + width / 2, rl_scores, width, label="ROUGE-L (Structure)", color="#f28e2b"
)

ax.set_ylabel("Score")
ax.set_title("Final Performance Comparison (N=64 Meetings)")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.set_ylim(0, 0.45)

plt.show()
