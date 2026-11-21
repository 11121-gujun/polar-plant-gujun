import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import scipy.stats as st

# ── 한글 폰트 ──
cands = ["AppleGothic","NanumGothic","NanumBarunGothic","Noto Sans CJK KR",
         "Malgun Gothic","Batang","Gulim"]
found = None
for c in cands:
    try:
        fp = font_manager.findfont(c, fallback_to_default=False)
        if fp and fp != font_manager.findfont("sans-serif"):
            found = c; break
    except Exception:
        continue
if not found:
    for f in font_manager.fontManager.ttflist:
        if "CJK" in f.name or "Nanum" in f.name:
            found = f.name; break
mpl.rcParams["font.family"] = found if found else "sans-serif"
mpl.rcParams["axes.unicode_minus"] = False

# ── 데이터 로드 ──
gro = pd.read_csv('/home/user/uploads/0_______________.csv', encoding='utf-8-sig')
env = pd.read_csv('/home/user/uploads/1______________.csv', encoding='utf-8-sig')

id_col    = "개체번호"
leaf_col  = "잎 수(장)"
fresh_col = "생중량(g)"
shoot_m   = "지상부질량(g)"
root_m    = "지하부질량(g)"

# 생중량 없으면 계산
if fresh_col not in gro.columns:
    gro[fresh_col] = gro[shoot_m] + gro[root_m]

print("=== 생육 데이터 열 ===")
print(list(gro.columns))
print(f"개체 수: {len(gro)}")

# ── 가설 H1: 잎 수 상위 25% vs 하위 20% 생중량 차이 ──
top_thr = gro[leaf_col].quantile(0.75)
bot_thr = gro[leaf_col].quantile(0.20)
top_ids = gro[gro[leaf_col] >= top_thr][id_col].tolist()
bot_ids = gro[gro[leaf_col] <= bot_thr][id_col].tolist()
top_v   = gro[gro[id_col].isin(top_ids)][fresh_col].dropna().values
bot_v   = gro[gro[id_col].isin(bot_ids)][fresh_col].dropna().values

mw_u, mw_p    = st.mannwhitneyu(top_v, bot_v, alternative="two-sided")
t_stat, t_p   = st.ttest_ind(top_v, bot_v, equal_var=False)
n1, n2        = len(top_v), len(bot_v)
pooled_sd     = np.sqrt(((n1-1)*top_v.std(ddof=1)**2 + (n2-1)*bot_v.std(ddof=1)**2)/(n1+n2-2))
cohen_d       = (top_v.mean() - bot_v.mean()) / pooled_sd

def es_label(d):
    d = abs(d)
    if d < 0.2:  return "매우 작음"
    if d < 0.5:  return "작음"
    if d < 0.8:  return "중간"
    return "큼"

print("\n=== H1 검정 결과 ===")
print(f"상위 기준: ≥{top_thr:.1f}장 → {n1}개체")
print(f"하위 기준: ≤{bot_thr:.1f}장 → {n2}개체")
print(f"평균 차이: {top_v.mean()-bot_v.mean():.4f} g")
print(f"Mann-Whitney U p = {mw_p:.5f}")
print(f"Welch t p = {t_p:.5f}")
print(f"Cohen's d = {cohen_d:.3f} ({es_label(cohen_d)})")

# ── 회귀식 ──
m, b = np.polyfit(gro[leaf_col].dropna(), gro[fresh_col].dropna(), 1)
r_val = gro[leaf_col].corr(gro[fresh_col])
print(f"\n회귀식: 생중량 = {m:.4f} × 잎수 + {b:.4f}")
print(f"r = {r_val:.4f}  R² = {r_val**2:.4f}")

# ════════════════════════════════════
# 시각물 1: 박스 그림 (H1 가설 검정)
# ════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(7.0, 5.8))

bp = ax1.boxplot(
    [top_v, bot_v],
    labels=[
        f"잎 수 상위 25%\n≥ {top_thr:.0f}장  (n={n1})",
        f"잎 수 하위 20%\n≤ {bot_thr:.0f}장  (n={n2})",
    ],
    patch_artist=True,
    widths=0.40,
    showmeans=True,
    meanprops=dict(marker="D", markerfacecolor="white",
                   markeredgecolor="black", markersize=7),
    medianprops=dict(linewidth=2.2, color="black"),
    flierprops=dict(marker="o", markerfacecolor="#bbb",
                    markeredgecolor="none", markersize=3.5, alpha=0.7),
)
for patch, color in zip(bp["boxes"], ["#4c72b0", "#e74c3c"]):
    patch.set_facecolor(color); patch.set_alpha(0.55); patch.set_linewidth(1.2)

ax1.set_title("잎 수 그룹별 생중량(g) 분포  —  H1 가설 검정",
              fontsize=13.5, pad=14)
ax1.set_ylabel("생중량 (g)", fontsize=11.5)
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.set_axisbelow(True)

# 상위 그룹 평균 주석
ax1.annotate(
    f"상위 평균: {top_v.mean():.2f} g\n상위 중앙값: {np.median(top_v):.2f} g",
    xy=(1, top_v.mean()),
    xytext=(1.42, top_v.max()*0.95),
    ha="left", va="top",
    arrowprops=dict(arrowstyle="->", color="#1a3a5c", lw=1.3),
    fontsize=9.5, color="#1a3a5c",
    bbox=dict(boxstyle="round,pad=0.4",
              fc="#e8f0fe", ec="#4c72b0", alpha=0.93),
)
# 하위 그룹 평균 주석
ax1.annotate(
    f"하위 평균: {bot_v.mean():.2f} g\n하위 중앙값: {np.median(bot_v):.2f} g",
    xy=(2, bot_v.mean()),
    xytext=(1.52, bot_v.max()*0.60),
    ha="left", va="bottom",
    arrowprops=dict(arrowstyle="->", color="#8b1a1a", lw=1.3),
    fontsize=9.5, color="#8b1a1a",
    bbox=dict(boxstyle="round,pad=0.4",
              fc="#fde8e8", ec="#e74c3c", alpha=0.93),
)

stat_txt = (
    f"Mann-Whitney U   p = {mw_p:.4f}  "
    f"→ {'유의함 ✓ (p<0.05)' if mw_p < 0.05 else '유의하지 않음'}\n"
    f"Welch t-test    p = {t_p:.4f}  "
    f"→ {'유의함 ✓ (p<0.05)' if t_p < 0.05 else '유의하지 않음'}\n"
    f"Cohen's d = {cohen_d:.2f}  ({es_label(cohen_d)} 효과)"
)
ax1.text(
    0.01, 0.98, stat_txt,
    transform=ax1.transAxes,
    fontsize=9.5, va="top", ha="left", family="monospace",
    bbox=dict(boxstyle="round,pad=0.5",
              fc="#fafafa", ec="#444", alpha=0.94, lw=1.2),
)
plt.tight_layout()
plt.savefig('/home/user/outputs/boxplot_h1_hypothesis.png', dpi=150, bbox_inches="tight")
plt.close(fig1)
print("✓ boxplot_h1_hypothesis.png 저장")

# ════════════════════════════════════
# 시각물 2: 산점도 (잎 수 vs 생중량, 회귀식 포함)
# ════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(7.6, 6.2))

valid_df = gro[[leaf_col, fresh_col, id_col]].dropna()
sc = ax2.scatter(
    valid_df[leaf_col], valid_df[fresh_col],
    s=75, alpha=0.72,
    c=valid_df[id_col], cmap="viridis",
    edgecolors="white", linewidth=0.6,
    zorder=3,
)
cbar = fig2.colorbar(sc, ax=ax2, label="개체번호", pad=0.02)
cbar.ax.tick_params(labelsize=9)

x_line = np.linspace(valid_df[leaf_col].min(), valid_df[leaf_col].max(), 200)
ax2.plot(x_line, m*x_line + b,
         color="#c0392b", linewidth=2.4, linestyle="--",
         label=f"추세선: y = {m:.3f}x + {b:.3f}",
         zorder=2)
ax2.legend(loc="upper left", fontsize=10.5, framealpha=0.92, edgecolor="#ccc")

ax2.set_title(f"{fresh_col} vs {leaf_col}  —  개체별 생육 산점도",
              fontsize=13.5, pad=14)
ax2.set_xlabel(leaf_col, fontsize=11.5)
ax2.set_ylabel(fresh_col, fontsize=11.5)
ax2.grid(True, alpha=0.25, linestyle="--")
ax2.set_axisbelow(True)

eq_box = (
    f"회귀식: y = {m:.3f}×x + {b:.3f}\n"
    f"상관계수 r = {r_val:.3f}   (R² = {r_val**2:.3f})\n"
    f"n = {len(valid_df)} 개체"
)
ax2.text(
    0.02, 0.97, eq_box,
    transform=ax2.transAxes,
    fontsize=10.5, va="top", ha="left",
    bbox=dict(boxstyle="round,pad=0.5",
              fc="white", ec="#555", alpha=0.95, lw=1.3),
    family="monospace",
)
plt.tight_layout()
plt.savefig('/home/user/outputs/scatter_leaf_vs_fresh.png', dpi=150, bbox_inches="tight")
plt.close(fig2)
print("✓ scatter_leaf_vs_fresh.png 저장")

# ════════════════════════════════════
# 시각물 3: 생육 변수 히트맵 (상관계수 행렬)
# ════════════════════════════════════
num_cols = ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)",
            "생중량(g)", "지상부질량(g)", "지하부질량(g)"]
corr = gro[num_cols].corr()
fig3, ax3 = plt.subplots(figsize=(7.0, 6.0))
im = ax3.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax3.set_xticks(range(len(num_cols)))
ax3.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=9.5)
ax3.set_yticks(range(len(num_cols)))
ax3.set_yticklabels(num_cols, fontsize=9.5)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        val = corr.iloc[i, j]
        ax3.text(j, i, f"{val:.2f}",
                 ha="center", va="center",
                 color="white" if abs(val) > 0.55 else "#222",
                 fontsize=10.5, fontweight="bold" if abs(val) > 0.7 else "normal")
ax3.set_title("생육 변수 간 피어슨 상관계수 행렬 (r)", fontsize=13, pad=12)
fig3.colorbar(im, ax=ax3, label="r (상관계수)")
plt.tight_layout()
plt.savefig('/home/user/outputs/correlation_heatmap.png', dpi=150, bbox_inches="tight")
plt.close(fig3)
print("✓ correlation_heatmap.png 저장")

print("\n=== 모두 저장 완료 ===")
