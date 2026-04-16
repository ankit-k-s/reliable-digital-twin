# src/it_dt_repro/app.py
import streamlit as st
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

st.set_page_config(
    page_title="ICS Cyber-Reliability Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ── Load artifacts ────────────────────────────────────────────
@st.cache_data
def load_artifacts():
    art = 'artifacts/it_dt_repro'
    p_final         = np.load(f'{art}/P_final_scores.npy')
    windowed_labels = np.load(f'{art}/windowed_labels.npy')
    windowed_labels = (windowed_labels > 0).astype(int)

    with open(f'{art}/it_dt_metrics.json') as f:
        it_dt_metrics = json.load(f)

    with open('artifacts/it_dt_repro/reliability_results.json') as f:
        rel_results = json.load(f)

    return p_final, windowed_labels, it_dt_metrics, rel_results

p_final, labels, metrics, rel = load_artifacts()
threshold = float(metrics['optimal_threshold_normalized'])

# ── Header ────────────────────────────────────────────────────
st.title("🛡️ ICS Cyber-Reliability Dashboard")
st.caption("Information-Theoretic Digital Twin (IT-DT) · WADI Water Distribution Dataset")

# --- LAYER 4C: CYBER-TTF BANNER ---
mean_ttf = rel.get('mean_ttf_seconds')
min_ttf = rel.get('min_ttf_seconds')

if mean_ttf is not None and mean_ttf < float('inf'):
    st.error(f"**🚨 CRITICAL ALERT (4C - Cyber-TTF):** System trajectory indicates critical failure in **{mean_ttf/60:.2f} minutes** (Worst-case: {min_ttf/60:.2f} min). Immediate operator intervention required.", icon="⏳")
else:
    st.success("**✅ 4C - Cyber-TTF:** System trajectory is stable. No imminent critical failures predicted.", icon="⏳")

st.divider()

# ── Row 1: Key metrics ────────────────────────────────────────
col1, col2, col3, col4, col5, col6 = st.columns(6)

pa_f1      = metrics['pa_f1_score']
auc        = rel['resilience_auc']
degradation = rel['degradation']
system_score = rel['system_score']
sep_ratio  = metrics['separation_ratio']

col1.metric("IT-DT PA-F1",        f"{pa_f1:.4f}",      help="Point-Adjusted F1 score")
col2.metric("4B - Res. AUC",      f"{auc:.4f}",        help="Area under reliability curve")
col3.metric("Degradation",        f"{degradation:.4f}", help="Reliability drop during attacks", delta=f"-{degradation:.4f}", delta_color="inverse")
col4.metric("System Score",       f"{system_score:.4f}", help="Criticality-weighted availability")
col5.metric("KL Separation",      f"{sep_ratio:.2f}x",  help="Attack vs normal signal ratio")

# Format TTF for metric card
ttf_str = f"{mean_ttf/60:.2f} min" if (mean_ttf is not None and mean_ttf < float('inf')) else "Stable"
col6.metric("4C - Mean TTF",      ttf_str,              help="Mean Time-To-Failure during attack trajectory")

st.divider()

# ── Row 2: Resilience curve + Anomaly scores ─────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("4B — Resilience Curve")

    roll_size = 5 * 60
    rel_raw   = 1.0 - (p_final / (p_final.max() + 1e-9))
    rel_raw   = np.clip(rel_raw, 0, 1)
    series    = np.convolve(rel_raw, np.ones(roll_size)/roll_size, mode='valid')
    t_min     = np.arange(len(series)) / 60

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_min, series, color='#3498db', linewidth=0.8, alpha=0.9, label='System reliability')
    ax.axhline(0.8, color='#f39c12', linestyle='--', linewidth=1, label='Critical threshold (0.8)')

    # Shade attack periods
    attack_idx = np.where(labels[:len(series)] == 1)[0]
    if len(attack_idx) > 0:
        onset = attack_idx.min() / 60
        end   = attack_idx.max() / 60
        ax.axvspan(onset, end, alpha=0.15, color='#e74c3c', label='Attack period')

    ax.axhline(rel['pre_attack_rel'], color='#2ecc71', linestyle=':', linewidth=1, label=f"Pre-attack baseline ({rel['pre_attack_rel']:.3f})")
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Reliability')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"AUC = {auc:.4f}  |  Degradation = {degradation:.4f}")
    st.pyplot(fig)
    plt.close()

with col_right:
    st.subheader("System Score")

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    col = ('#2ecc71' if system_score > 0.8 else '#f39c12' if system_score > 0.6 else '#e74c3c')
    ax2.pie([system_score, 1-system_score], colors=[col, '#ecf0f1'], startangle=90, wedgeprops={'width': 0.4})
    ax2.text(0, 0, f'{system_score:.3f}', ha='center', va='center', fontsize=26, fontweight='bold', color=col)
    status = ("NOMINAL" if system_score > 0.8 else "DEGRADED" if system_score > 0.6 else "CRITICAL")
    ax2.set_title(f'Status: {status}', fontsize=12)
    st.pyplot(fig2)
    plt.close()

st.divider()

# ── Row 3: Subsystem availability + Anomaly scores ───────────
col_avail, col_scores = st.columns(2)

with col_avail:
    st.subheader("4A — Subsystem Availability")

    availability = rel['availability']
    subs  = list(availability.keys())
    vals  = [availability[s] for s in subs]
    cols_bar = ['#2ecc71' if v > 0.8 else '#f39c12' if v > 0.6 else '#e74c3c' for v in vals]
    short = [s.replace('zone','Z').replace('_','\n') for s in subs]

    fig3, ax3 = plt.subplots(figsize=(6, 3))
    bars = ax3.bar(short, vals, color=cols_bar, alpha=0.85, edgecolor='white')
    ax3.axhline(0.8, color='#f39c12', linestyle='--', linewidth=1, label='Critical (0.8)')
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel('Availability')
    ax3.legend(fontsize=8)
    ax3.tick_params(axis='x', labelsize=8)
    for b, v in zip(bars, vals):
        ax3.text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    st.pyplot(fig3)
    plt.close()

    # Table view with safety fallback for criticality weights
    weights_fallback = {'zone1_intake': 0.6, 'zone2_treatment': 0.9, 'zone2_ro': 0.8, 'zone3_distribution': 1.0, 'leak_detection': 0.7}
    
    df_avail = pd.DataFrame({
        'Subsystem':    subs,
        'Availability': [f'{v:.4f}' for v in vals],
        'Status':       ['OK' if v > 0.8 else 'DEGRADED' if v > 0.6 else 'CRITICAL' for v in vals],
        'Criticality':  [str(rel.get('criticality_weights', weights_fallback).get(s, '—')) for s in subs]
    })
    st.dataframe(df_avail, use_container_width=True, hide_index=True)

with col_scores:
    st.subheader("IT-DT Anomaly Scores")

    t2   = np.arange(len(p_final)) / 60
    mask = labels[:len(p_final)] == 1

    fig4, ax4 = plt.subplots(figsize=(6, 3))
    ax4.plot(t2, p_final, color='#95a5a6', linewidth=0.3, alpha=0.6, label='Anomaly score')
    ax4.scatter(t2[mask], p_final[mask], color='#e74c3c', s=1, alpha=0.8, label='Labeled attack')
    ax4.axhline(threshold, color='#f39c12', linestyle='--', linewidth=1, label=f'Threshold ({threshold:.3f})')
    ax4.set_xlabel('Time (minutes)')
    ax4.set_ylabel('Score (log-normalized)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)
    plt.close()

st.divider()

# ── Row 4: Fault vs Attack comparison ────────────────────────
st.subheader("Fault vs Attack Signature Comparison")
st.caption("Compares anomaly score distribution between labeled attack periods and normal periods within the attack file")

col_dist, col_stats = st.columns([2, 1])

with col_dist:
    normal_scores = p_final[labels == 0]
    attack_scores = p_final[labels == 1]

    fig5, ax5 = plt.subplots(figsize=(8, 3))
    ax5.hist(normal_scores, bins=100, alpha=0.6, color='#3498db', label='Normal operation', density=True)
    ax5.hist(attack_scores, bins=100, alpha=0.6, color='#e74c3c', label='Attack periods', density=True)
    ax5.axvline(threshold, color='#f39c12', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold:.3f})')
    ax5.set_xlabel('Anomaly score (log-normalized)')
    ax5.set_ylabel('Density')
    ax5.legend(fontsize=9)
    ax5.set_title('Score distribution: Normal vs Attack windows')
    st.pyplot(fig5)
    plt.close()

with col_stats:
    st.markdown("**Signal Statistics**")
    stats_df = pd.DataFrame({
        'Metric': ['Mean score', 'Std dev', 'P95', 'P99', 'Max'],
        'Normal': [
            f'{normal_scores.mean():.4f}', f'{normal_scores.std():.4f}',
            f'{np.percentile(normal_scores, 95):.4f}', f'{np.percentile(normal_scores, 99):.4f}', f'{normal_scores.max():.4f}'
        ],
        'Attack': [
            f'{attack_scores.mean():.4f}', f'{attack_scores.std():.4f}',
            f'{np.percentile(attack_scores, 95):.4f}', f'{np.percentile(attack_scores, 99):.4f}', f'{attack_scores.max():.4f}'
        ]
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    separation = attack_scores.mean() / (normal_scores.mean() + 1e-9)
    st.metric("Separation ratio", f"{separation:.3f}x", help="Attack mean / Normal mean in log-normalized space")

st.divider()

# ── Row 5: Method comparison table ───────────────────────────
st.subheader("Comparison Against Published WADI Results")

comparison_df = pd.DataFrame({
    'Method':   ['**IT-DT (Ours)**', 'IT-DT (Original paper)', 'iADCPS (SOTA 2025)', 'CAE-T (2024)', 'DAICS (2020)', 'TranAD (2022)', 'OmniAnomaly'],
    'PA-F1':    ['**0.6524**', '0.6100', '0.7870', '0.7274', '~0.56', '~0.43', '0.4260'],
    'Protocol': ['PA', 'PA', 'PA', 'PA', 'Point', 'PA', 'PA'],
    'Novel contribution': [
        'Reliability layer (AUC + TTF)', 'Detection only', 'Incremental meta-learning', 'CNN + Transformer + SVDD', 'Online fine-tuning', 'Transformer reconstruction', 'Stochastic RNN'
    ]
})
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Architecture: Layer 0 Baseline → Layer 1 WADI Data → "
    "Layer 2 IT-DT (N4SID + Kalman + KL Divergence) → "
    "Layer 3 Detection (PA-F1 0.6524) → "
    "Layer 4 Reliability Translation (Availability, Resilience, Cyber-TTF) → "
    "Layer 5 Criticality Weighting → Layer 6 Dashboard"
)