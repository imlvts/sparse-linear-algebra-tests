import csv
import matplotlib.pyplot as plt

ns, overheads, pcts = [], [], []
with open("btree_overhead.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        n = int(row["n"])
        il = int(row["internal_len"])
        ns.append(n)
        overheads.append(il)
        pcts.append(100.0 * il / n if n > 0 else 0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(ns, overheads, linewidth=0.5, color="steelblue")
ax1.set_ylabel("Internal nodes (absolute)")
ax1.set_title("DenseBTree overhead vs data length (KEYS_PER_NODE=16)")
ax1.grid(True, alpha=0.3)

ax2.plot(ns, pcts, linewidth=0.5, color="coral")
ax2.set_ylabel("Overhead (%)")
ax2.set_xlabel("Data length (n)")
ax2.set_ylim(0, max(pcts) * 1.05)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("btree_overhead.png", dpi=150)
print("Saved btree_overhead.png")
