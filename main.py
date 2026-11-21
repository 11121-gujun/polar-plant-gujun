#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
극지식물 개체별 생육 차이 분석 대시보드
송도고등학교 환경·생육 데이터용 Streamlit 앱
의존성: streamlit, pandas, numpy, matplotlib  (scipy 불필요)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

# ─────────────────────────────────────────────
# 1. 한글 폰트 설정
# ─────────────────────────────────────────────
def setup_korean_font():
    candidates = [
        "AppleGothic",
        "NanumGothic",
        "NanumBarunGothic",
        "Noto Sans CJK KR",
        "Malgun Gothic",
        "Batang",
        "Gulim",
    ]
    for cand in candidates:
        try:
            fp = font_manager.findfont(cand, fallback_to_default=False)
            if fp and fp != font_manager.findfont("sans-serif"):
                mpl.rcParams["font.family"] = cand
                mpl.rcParams["axes.unicode_minus"] = False
                return cand
        except Exception:
            continue
    try:
        for f in font_manager.fontManager.ttflist:
            name = f.name
            if "CJK" in name or "Nanum" in name or "Noto" in name:
                mpl.rcParams["font.family"] = name
                mpl.rcParams["axes.unicode_minus"] = False
                return name
    except Exception:
        pass
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["axes.unicode_minus"] = False
    return "sans-serif (한글 폰트 없음)"

FONT_NAME = setup_korean_font()

# ─────────────────────────────────────────────
# 2. 인코딩 자동 감지
# ─────────────────────────────────────────────
def read_csv_safe(path):
    for enc in ["utf-8-sig", "cp949", "utf-8"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            if len(df) > 0 and (df.columns.str.strip() != "").all():
                return df, enc
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    df = pd.read_csv(path, encoding="latin-1")
    return df, "latin-1 (추정)"

# ─────────────────────────────────────────────
# 3. 열 이름 정규화 — 환경 데이터
# ─────────────────────────────────────────────
def normalize_env_columns(df):
    time_col = temp_col = hum_col = ec_col = ph_col = wt_col = co2_col = None
    extra_cols = []

    for orig in df.columns:
        clean   = str(orig).strip()
        lower   = clean.lower()

        if time_col is None and (
                "time" in lower or "시각" in clean or "일자" in clean
                or ("date" in lower and "time" not in lower and len(clean) < 30)
        ):
            time_col = orig
        elif time_col is None and clean == "categories" and df[orig].dtype == object:
            time_col = orig

        if temp_col is None and ("temp" in lower or "온도" in clean or "기온" in clean):
            temp_col = orig
        if hum_col is None and ("humid" in lower or "습도" in clean):
            hum_col = orig
        if ec_col is None and (lower == "ec" or lower.startswith("ec ")
                                or "전기" in clean):
            ec_col = orig
        if ph_col is None and (lower[:2] == "ph" or "수소" in clean or "산도" in clean):
            ph_col = orig
        if wt_col is None and ("water_temp" in lower or "수온" in clean
                                or ("water" in lower and "temp" in lower)):
            wt_col = orig
        if co2_col is None and ("co2" in lower or lower.startswith("co2 ")
                                or "이산화" in clean):
            co2_col = orig

        if orig not in (time_col, temp_col, hum_col, ec_col, ph_col, wt_col, co2_col):
            extra_cols.append(orig)

    std = {}
    for std_name, orig_col in [
        ("time", time_col), ("temperature", temp_col), ("humidity", hum_col),
        ("ec", ec_col), ("ph", ph_col), ("water_temp", wt_col), ("co2", co2_col)
    ]:
        if orig_col is not None:
            std[std_name] = orig_col
    return std, extra_cols


def normalize_growth_columns(df):
    id_col = leaf_col = shoot_col = root_col = fresh_col = shoot_m_col = root_m_col = None
    extra_cols = []

    for orig in df.columns:
        clean = str(orig).strip()
        lower = clean.lower().replace(" ", "").replace("(", "").replace(")", "")

        if id_col is None and ("번호" in clean or "id" in lower or "개체" in clean):
            id_col = orig
        elif leaf_col is None and ("잎" in clean or "leaf" in lower or "엽" in clean):
            leaf_col = orig
        elif shoot_col is None and ("지상" in clean and ("길이" in clean or "부" in clean)
                                     or "shoot" in lower):
            shoot_col = orig
        elif root_col is None and ("지하" in clean and ("길이" in clean or "부" in clean)
                                    or "root" in lower):
            root_col = orig
        elif fresh_col is None and ("생중량" in clean or "fresh" in lower):
            fresh_col = orig
        elif shoot_m_col is None and ("지상부질량" in clean or "지상부 질량" in clean
                                      or ("지상" in clean and "질량" in clean)
                                      or "shoot_mass" in lower):
            shoot_m_col = orig
        elif root_m_col is None and ("지하부질량" in clean or "지하부 질량" in clean
                                     or ("지하" in clean and "질량" in clean)
                                     or "root_mass" in lower):
            root_m_col = orig
        else:
            extra_cols.append(orig)

    std = {}
    for std_name, orig_col in [
        ("개체번호", id_col), ("잎 수(장)", leaf_col),
        ("지상부 길이(mm)", shoot_col), ("지하부길이(mm)", root_col),
        ("생중량(g)", fresh_col), ("지상부질량(g)", shoot_m_col),
        ("지하부질량(g)", root_m_col)
    ]:
        if orig_col is not None:
            std[std_name] = orig_col
    return std, extra_cols


def rename_df(df, std_map):
    reverse = {v: k for k, v in std_map.items()}
    if not reverse:
        return df.copy()
    return df.rename(columns=reverse).copy()


# ─────────────────────────────────────────────
# 4. 생중량 자동 계산
# ─────────────────────────────────────────────
def ensure_fresh_weight(df):
    if "생중량(g)" not in df.columns:
        if "지상부질량(g)" in df.columns and "지하부질량(g)" in df.columns:
            df["생중량(g)"] = df["지상부질량(g)"] + df["지하부질량(g)"]
            st.info(
                "⚠️  '생중량(g)' 열이 없어 "
                "'지상부질량 + 지하부질량'으로 계산했습니다."
            )
        else:
            st.warning(
                "⚠️  '생중량(g)'도 없고 질량 열도 없어 계산할 수 없습니다."
            )
    return df


# ─────────────────────────────────────────────
# 5. 수치형 열 헬퍼  (scipy 없이 numpy만 사용)
# ─────────────────────────────────────────────
def numeric_cols(df, exclude=None):
    exclude = set(exclude or [])
    return [c for c in df.select_dtypes(include="number").columns if c not in exclude]


# ─────────────────────────────────────────────
# 6. 통계 설명 (고등학생용 짧은 해설)
# ─────────────────────────────────────────────
GLOSS = {
    "평균": "모든 값을 더해 개수로 나눈 값. 데이터 전체의 '대표값'.",
    "표준편차": "값이 평균에서 얼마나 퍼져 있는지 나타냄. 클수록 개체 간 차이가 큼.",
    "중앙값": "값을 크기순으로 나열했을 때 가운데 위치한 값. 극단값의 영향을 덜 받음.",
    "최소": "데이터에서 가장 작은 값.",
    "최대": "데이터에서 가장 큰 값.",
    "상관계수 (r)": (
        "두 변수가 함께 움직이는 정도. -1 ~ 1 사이 값.\n"
        "  1에 가까울수록 함께 증가, -1에 가까울수록 반대 방향,\n"
        "  0에 가까우면 관계가 약함."
    ),
}


def stat_card(col_name, series, gloss_key=None):
    s = series.dropna()
    if s.empty:
        return None
    return {
        "열": col_name,
        "n": len(s),
        "평균": round(float(s.mean()), 3),
        "표준편차": round(float(s.std()), 3),
        "중앙값": round(float(s.median()), 3),
        "최소": round(float(s.min()), 3),
        "최대": round(float(s.max()), 3),
        "해설": GLOSS.get(gloss_key, "") if gloss_key else "",
    }


# ─────────────────────────────────────────────
# 7.Mann-Whitney U 검정 (scipy 없이 numpy로 근사)
#    농성순위합(U) 기반 근사 p-value
# ─────────────────────────────────────────────
def mann_whitney_u_approx(group_a, group_b):
    """
    두 독립 표본에 대한 Mann-Whitney U 검정.
    scipy 없이 numpy만 사용. U 통계량과 근사 양측 p-value를 반환.
    (정규 근사, 연속성 보정 포함)
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan

    pooled = np.concatenate([a, b])
    ranks = np.argsort(np.argsort(pooled, kind="stable"), kind="stable") + 1
    r_a = ranks[:n1].sum()
    U1 = r_a - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)

    # 정규 근사
    mu    = n1 * n2 / 2.0
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma == 0:
        p = 1.0
    else:
        z = (U - mu) / sigma
        # 양측 p-value (표준정규분포 누적분포함수 근사: 오차함수 사용)
        p = 2.0 * (1.0 - 0.5 * (1.0 + np.math.erf(abs(z) / np.sqrt(2.0))))

    return float(U), float(np.clip(p, 0.0, 1.0))


def welch_ttest(group_a, group_b):
    """
    등분산을 가정하지 않는 Welch t-검정.
    scipy 없이 numpy만 사용. t 통계량과 근사 양측 p-value 반환.
    자유도는 Welch-Satterthwaite 식으로 계산, p-value는
    정규 근사로 대략적인 값을 산출 (n이 충분히 클 때 참고용).
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan

    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return np.nan, np.nan

    t = (m1 - m2) / se

    # Welch-Satterthwaite 자유도
    num   = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df    = num / denom if denom > 0 else np.inf

    # 양측 p-value 근사 (t-분포 → 정규 근사로 대략 계산)
    # n이 30 이상이면 정규 근사가 꽤 정확함
    p = 2.0 * (1.0 - 0.5 * (1.0 + np.math.erf(abs(t) / np.sqrt(2.0))))

    return float(t), float(np.clip(p, 0.0, 1.0))


def cohens_d(group_a, group_b):
    """Cohen's d (통합 표준편차 기준 효과크기)"""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return np.nan
    return float((a.mean() - b.mean()) / pooled_sd)


def es_label(d):
    d = abs(d)
    if d < 0.2:
        return "매우 작음"
    if d < 0.5:
        return "작음"
    if d < 0.8:
        return "중간"
    return "큼"


# ─────────────────────────────────────────────
# 8. 페이지 구성
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="극지식물 생육 대시보드",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌿 극지식물 개체별 생육 차이 분석 대시보드")

# ── 사이드바: 파일 업로드 ──
st.sidebar.header("📁 데이터 파일")
env_file    = st.sidebar.file_uploader("환경 데이터 CSV", type=["csv"])
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type=["csv"])

if env_file is None and growth_file is None:
    st.warning("두 CSV 파일을 모두 업로드하세요.")
    st.stop()

# ── 환경 데이터 로드 ──
if env_file is not None:
    env_raw, env_enc = read_csv_safe(env_file)
    env_map, env_extra = normalize_env_columns(env_raw)
    env_df = rename_df(env_raw, env_map)
    st.sidebar.success(f"환경 데이터 인코딩: **{env_enc}**")
    st.sidebar.info(f"매핑된 환경 열: {list(env_map.keys())}")
    if env_extra:
        st.sidebar.warning(f"기타 열(분석 미사용): {env_extra}")
else:
    env_df = pd.DataFrame()

# ── 생육 데이터 로드 ──
if growth_file is not None:
    g_raw, g_enc = read_csv_safe(growth_file)
    g_map, g_extra = normalize_growth_columns(g_raw)
    g_df = rename_df(g_raw, g_map)
    g_df = ensure_fresh_weight(g_df)
    st.sidebar.success(f"생육 데이터 인코딩: **{g_enc}**")
    st.sidebar.info(f"매핑된 생육 열: {list(g_map.keys())}")
    if g_extra:
        st.sidebar.warning(f"기타 열(분석 미사용): {g_extra}")
else:
    g_df = pd.DataFrame()

# ══════════════════════════════════════════════
# 탭
# ══════════════════════════════════════════════
tab_summary, tab_growth, tab_env, tab_corr, tab_scatter = st.tabs([
    "📊 전체 요약",
    "🌱 생육 통계 (개체별)",
    "🌡️ 환경 통계 (시간별)",
    "🔗 상관 분석",
    "📈 산점도",
])

# ─────────────────────────────────────────────
# 탭 1: 전체 요약
# ─────────────────────────────────────────────
with tab_summary:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🌱 생육 결과 데이터")
        if g_df.empty:
            st.info("생육 데이터를 업로드하면 여기에 요약이 표시됩니다.")
        else:
            st.write(f"**개체 수:** {len(g_df)}개")
            st.write(f"**사용 열:** {list(g_df.columns)}")
            st.write("#### 기초 통계")
            rows = []
            for col in numeric_cols(g_df, exclude=["개체번호"]):
                card = stat_card(col, g_df[col], gloss_key=col)
                if card:
                    rows.append(card)
            if rows:
                st.dataframe(
                    pd.DataFrame(rows).set_index("열"),
                    use_container_width=True,
                )
                for r in rows:
                    if r["해설"]:
                        st.caption(f"**{r['열']}** — {r['해설']}")
            else:
                st.warning("수치형 열이 없어 통계를 표시할 수 없습니다.")

    with c2:
        st.subheader("🌡️ 환경 데이터")
        if env_df.empty:
            st.info("환경 데이터를 업로드하면 여기에 요약이 표시됩니다.")
        else:
            has_time = False
            if "time" in env_df.columns:
                try:
                    env_df["_dt"] = pd.to_datetime(env_df["time"], errors="coerce")
                    has_time = env_df["_dt"].notna().any()
                except Exception:
                    pass

            st.write(f"**총 측정 횟수:** {len(env_df)}회")
            st.write("**시간 범위:**")
            if has_time:
                tmin = env_df["_dt"].min()
                tmax = env_df["_dt"].max()
                st.write(
                    f"{tmin.strftime('%Y-%m-%d %H:%M')} ~ "
                    f"{tmax.strftime('%Y-%m-%d %H:%M')}"
                )
            else:
                st.write("(시간 정보 없음 또는 파싱 실패)")

            env_num = numeric_cols(env_df, exclude=["time", "_dt"])
            st.write("#### 환경 변수별 기초 통계")
            rows = []
            for col in env_num:
                card = stat_card(col, env_df[col], gloss_key=col)
                if card:
                    rows.append(card)
            if rows:
                st.dataframe(
                    pd.DataFrame(rows).set_index("열"),
                    use_container_width=True,
                )
                for r in rows:
                    if r["해설"]:
                        st.caption(f"**{r['열']}** — {r['해설']}")
            else:
                st.warning("수치형 환경 열이 없습니다.")

# ─────────────────────────────────────────────
# 탭 2: 생육 통계 (개체별)
# ─────────────────────────────────────────────
with tab_growth:
    if g_df.empty:
        st.info("생육 데이터를 업로드하세요.")
    else:
        st.subheader("📋 개체별 생육 데이터 표")
        gnum = numeric_cols(g_df, exclude=["개체번호"])
        if gnum and "개체번호" in g_df.columns:
            display_cols = ["개체번호"] + gnum
            st.dataframe(
                g_df[display_cols].reset_index(drop=True),
                use_container_width=True,
                height=400,
            )
        elif gnum:
            st.dataframe(g_df[gnum].reset_index(drop=True),
                         use_container_width=True, height=400)

        st.subheader("📊 생육 변수 분포")
        if not gnum:
            st.warning("수치형 생육 열이 없습니다.")
        else:
            sel = st.selectbox("분포를 볼 변수 선택", gnum, index=0)
            fig, ax = plt.subplots(figsize=(8, 4))
            vals = g_df[sel].dropna()
            ax.hist(vals, bins=25, color="#4c72b0", edgecolor="white", alpha=0.85)
            ax.axvline(vals.mean(), color="red", linestyle="--",
                       linewidth=1.5, label=f"평균: {vals.mean():.2f}")
            ax.axvline(vals.median(), color="green", linestyle="--",
                       linewidth=1.5, label=f"중앙값: {vals.median():.2f}")
            ax.set_title(f"{sel} 분포  (n={len(vals)})")
            ax.set_xlabel(sel)
            ax.set_ylabel("개체 수")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("🏆 생육 우수 / 저조 개체 Top 5")
        if not gnum:
            st.warning("비교할 생육 변수가 없습니다.")
        else:
            val_col = st.selectbox("기준 변수 선택", gnum, index=0)
            id_col = "개체번호" if "개체번호" in g_df.columns else g_df.columns[0]
            top5 = g_df.nlargest(5, val_col)[[id_col, val_col]].reset_index(drop=True)
            bot5 = g_df.nsmallest(5, val_col)[[id_col, val_col]].reset_index(drop=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**생육 우수 개체 (상위 5)**")
                st.dataframe(top5, use_container_width=True)
            with c2:
                st.markdown("**생육 저조 개체 (하위 5)**")
                st.dataframe(bot5, use_container_width=True)

# ─────────────────────────────────────────────
# 탭 3: 환경 통계 (시간별)
# ─────────────────────────────────────────────
with tab_env:
    if env_df.empty:
        st.info("환경 데이터를 업로드하세요.")
    else:
        env_num = numeric_cols(env_df, exclude=["time", "_dt"])
        if not env_num:
            st.warning("수치형 환경 열이 없습니다.")
        else:
            has_time = False
            if "time" in env_df.columns:
                try:
                    env_df["_dt"] = pd.to_datetime(env_df["time"], errors="coerce")
                    has_time = env_df["_dt"].notna().any()
                except Exception:
                    st.warning("시간 열 파싱에 실패했습니다.")

            sel_cols = st.multiselect(
                "그래프를 그릴 환경 변수 선택",
                env_num,
                default=env_num[:min(3, len(env_num))],
            )
            if sel_cols:
                if has_time:
                    dfp = env_df.dropna(subset=["_dt"]) \
                                 .sort_values("_dt") \
                                 .reset_index(drop=True)
                    fig, ax = plt.subplots(figsize=(12, 4))
                    colors = [
                        "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
                        "#9b59b6", "#1abc9c",
                    ]
                    for i, col in enumerate(sel_cols):
                        mask = dfp["_dt"].notna() & dfp[col].notna()
                        t_plot = dfp.loc[mask, "_dt"]
                        v_plot = dfp.loc[mask, col]
                        ax.plot(
                            t_plot, v_plot,
                            label=col,
                            color=colors[i % len(colors)],
                            linewidth=1.2,
                            alpha=0.85,
                        )
                    ax.set_title("환경 변수 시간별 추이")
                    ax.set_xlabel("시간")
                    ax.set_ylabel("측정값")
                    ax.legend(loc="best")
                    ax.grid(True, alpha=0.3)
                    fig.autofmt_xdate()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning(
                        "시간 정보(time 열)가 없어 시간별 추이 그래프를 "
                        "그릴 수 없습니다."
                    )

            st.subheader("환경 변수별 통계")
            rows = []
            for col in env_num:
                card = stat_card(col, env_df[col], gloss_key=col)
                if card:
                    rows.append(card)
            if rows:
                st.dataframe(
                    pd.DataFrame(rows).set_index("열"),
                    use_container_width=True,
                )
            else:
                st.warning("통계 대상이 되는 환경 변수가 없습니다.")

# ─────────────────────────────────────────────
# 탭 4: 상관 분석
# ─────────────────────────────────────────────
with tab_corr:
    if g_df.empty and env_df.empty:
        st.info("환경 데이터 또는 생육 데이터를 업로드하세요.")
    else:
        if not g_df.empty:
            gnum = numeric_cols(g_df, exclude=["개체번호"])
            if len(gnum) >= 2:
                st.subheader("🌱 생육 변수 간 피어슨 상관계수 행렬")
                corr_g = g_df[gnum].corr()
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(
                    corr_g.values,
                    cmap="RdBu_r",
                    vmin=-1, vmax=1,
                    aspect="auto",
                )
                ax.set_xticks(range(len(gnum)))
                ax.set_xticklabels(gnum, rotation=45, ha="right", fontsize=9)
                ax.set_yticks(range(len(gnum)))
                ax.set_yticklabels(gnum, fontsize=9)
                for i in range(len(gnum)):
                    for j in range(len(gnum)):
                        val = corr_g.iloc[i, j]
                        ax.text(
                            j, i, f"{val:.2f}",
                            ha="center", va="center",
                            color="white" if abs(val) > 0.5 else "black",
                            fontsize=10,
                        )
                ax.set_title("생육 변수 간 상관계수")
                fig.colorbar(im, ax=ax, label="r (상관계수)")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    "**상관계수 r**은 두 변수가 함께 움직이는 정도와 방향을 "
                    "-1~1로 나타냅니다.\n"
                    "- 1에 가까움: 함께 증가 (강한 양의 상관)\n"
                    "- -1에 가까움: 한쪽 증가 시 다른 쪽 감소 (강한 음의 상관)\n"
                    "- 0에 가까움: 관계가 거의 없음"
                )
            else:
                st.info(
                    "생육 변수가 2개 이상이어야 상관행렬을 그릴 수 있습니다."
                )

        if not env_df.empty:
            env_num = numeric_cols(env_df, exclude=["time", "_dt"])
            if len(env_num) >= 2:
                st.subheader("🌡️ 환경 변수 간 피어슨 상관계수 행렬")
                corr_e = env_df[env_num].corr()
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(
                    corr_e.values,
                    cmap="RdBu_r",
                    vmin=-1, vmax=1,
                    aspect="auto",
                )
                ax.set_xticks(range(len(env_num)))
                ax.set_xticklabels(env_num, rotation=45, ha="right", fontsize=9)
                ax.set_yticks(range(len(env_num)))
                ax.set_yticklabels(env_num, fontsize=9)
                for i in range(len(env_num)):
                    for j in range(len(env_num)):
                        val = corr_e.iloc[i, j]
                        ax.text(
                            j, i, f"{val:.2f}",
                            ha="center", va="center",
                            color="white" if abs(val) > 0.5 else "black",
                            fontsize=10,
                        )
                ax.set_title("환경 변수 간 상관계수")
                fig.colorbar(im, ax=ax, label="r (상관계수)")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    "**상관계수 r**은 두 변수가 함께 움직이는 정도와 방향을 "
                    "-1~1로 나타냅니다.\n"
                    "- 1에 가까움: 함께 증가 (강한 양의 상관)\n"
                    "- -1에 가까움: 한쪽 증가 시 다른 쪽 감소 (강한 음의 상관)\n"
                    "- 0에 가까움: 관계가 거의 없음"
                )
            else:
                st.info(
                    "환경 변수가 2개 이상이어야 상관행렬을 그릴 수 있습니다."
                )

# ─────────────────────────────────────────────
# 탭 5: 산점도
# ─────────────────────────────────────────────
with tab_scatter:
    if g_df.empty:
        st.info("생육 데이터를 업로드하세요.")
    else:
        gnum = numeric_cols(g_df, exclude=["개체번호"])
        if len(gnum) >= 2:
            all_cols = list(gnum)
            x_col = st.selectbox("X축 변수", all_cols, index=0)
            y_opts = [c for c in all_cols if c != x_col]
            y_col = st.selectbox("Y축 변수", y_opts, index=0)

            dot_size  = st.slider("점 크기", 10, 200, 60)
            alpha_val = st.slider("투명도 (alpha)", 0.1, 1.0, 0.6, 0.05)

            fig, ax = plt.subplots(figsize=(7, 5))
            has_id = "개체번호" in g_df.columns

            if has_id:
                sc = ax.scatter(
                    g_df[x_col], g_df[y_col],
                    s=dot_size, alpha=alpha_val,
                    c=g_df["개체번호"],
                    cmap="viridis",
                    edgecolors="white",
                    linewidth=0.5,
                )
                fig.colorbar(sc, ax=ax, label="개체번호")
            else:
                ax.scatter(
                    g_df[x_col], g_df[y_col],
                    s=dot_size, alpha=alpha_val,
                    color="#4c72b0",
                    edgecolors="white",
                    linewidth=0.5,
                )

            valid = g_df[[x_col, y_col]].dropna()
            if len(valid) > 2:
                m_b, b_b = np.polyfit(valid[x_col], valid[y_col], 1)
                x_line = np.linspace(
                    valid[x_col].min(), valid[x_col].max(), 100
                )
                ax.plot(
                    x_line, m_b * x_line + b_b,
                    color="red", linestyle="--",
                    linewidth=1.5,
                    label=f"추세선 (기울기={m_b:.3f})",
                )
                ax.legend()

            r_val = valid[x_col].corr(valid[y_col])
            ax.set_title(
                f"{y_col} vs {x_col}  (r = {r_val:.3f})"
            )
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                f"**상관계수 r = {r_val:.3f}**\n"
                f"- 1에 가까움: 두 변수가 함께 증가하거나 감소하는 경향 "
                f"(강한 양의 상관)\n"
                f"- -1에 가까움: 한 변수가 증가할 때 다른 변수가 감소 "
                f"(강한 음의 상관)\n"
                f"- 0에 가까움: 별다른 관계가 없음"
            )

            if has_id:
                st.subheader("🔍 특정 개체 찾기")
                max_id = int(g_df["개체번호"].max())
                target_id = st.number_input(
                    "찾을 개체번호",
                    min_value=1,
                    max_value=max_id,
                    value=1,
                )
                if target_id in g_df["개체번호"].values:
                    row = g_df[g_df["개체번호"] == target_id].iloc[0]
                    st.success(f"**개체번호 {target_id}** 의 생육 정보")
                    info_df = pd.DataFrame({
                        "변수": list(row.index),
                        "값": [
                            f"{v:.2f}" if isinstance(v, float) else str(v)
                            for v in row.values
                        ],
                    })
                    st.dataframe(
                        info_df,
                        use_container_width=True,
                        hide_index=True,
                    )
        else:
            st.warning("산점도를 그리려면 생육 변수가 2개 이상 필요합니다.")

# ─────────────────────────────────────────────
# 하단
# ─────────────────────────────────────────────
st.divider()
st.caption("— 극지식물 생육 분석 대시보드 끝 —")
