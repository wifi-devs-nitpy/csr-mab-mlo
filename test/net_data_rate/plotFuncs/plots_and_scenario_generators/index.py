import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------- FIGURE SETUP ----------
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# ---------- HELPER FUNCTIONS ----------
def draw_box(x, y, w, h, text, color):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        edgecolor=color,
        facecolor=color,
        alpha=0.15,
        linewidth=2
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10)

def draw_arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=2))

# ---------- STEP 1: AP GROUP ----------
draw_box(0.1, 0.75, 0.8, 0.15,
         "Level 1: AP Group Selection\nSelect subset of APs\nExample: {AP-A, AP-B}",
         "blue")

# ---------- STEP 2: STATION ----------
draw_box(0.1, 0.55, 0.8, 0.15,
         "Level 2: Station Selection\nEach AP selects one STA\nAP-A → SA-2, AP-B → SB-3",
         "green")

# ---------- STEP 3: LINK ----------
draw_box(0.1, 0.35, 0.8, 0.15,
         "Level 3: Link Selection\nChoose subset of links (3 links only)\nAP-A → {L1, L3}, AP-B → {L2}",
         "orange")

# ---------- STEP 4: TX POWER ----------
draw_box(0.1, 0.15, 0.8, 0.15,
         "Level 4: Power Selection\nAssign power per link\nL1 → P2, L2 → P1, L3 → P3",
         "purple")

# ---------- ARROWS ----------
draw_arrow(0.5, 0.75, 0.5, 0.70)
draw_arrow(0.5, 0.55, 0.5, 0.50)
draw_arrow(0.5, 0.35, 0.5, 0.30)

# ---------- TITLE ----------
ax.text(0.5, 0.95,
        "Four-Level Hierarchical MAB for C-SR",
        ha='center', fontsize=16, fontweight='bold')

plt.show()