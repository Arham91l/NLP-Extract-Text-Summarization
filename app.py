import streamlit as st
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from summarizer import cosine_tfidf_summary, get_sentence_scores, compression_stats

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SummarAI — Text Summarizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root & Background ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d0d1a 40%, #0a0f1a 100%);
    min-height: 100vh;
}

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Hero Header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00f5a0, #00d9f5, #a855f7, #f97316);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}
.hero-sub {
    font-size: 1.05rem;
    color: #8892a4;
    font-weight: 300;
    letter-spacing: 0.5px;
}
.hero-badge {
    display: inline-block;
    background: rgba(0, 245, 160, 0.12);
    border: 1px solid rgba(0, 245, 160, 0.3);
    color: #00f5a0;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 12px;
    border-radius: 20px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Glowing divider ── */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #00f5a0, #a855f7, transparent);
    margin: 1rem 0 2rem;
    opacity: 0.5;
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 1.2rem 0;
}
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: rgba(0,245,160,0.3); }
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00f5a0, #00d9f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    font-size: 0.72rem;
    color: #8892a4;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}
.metric-card.orange .metric-val {
    background: linear-gradient(135deg, #f97316, #fbbf24);
    -webkit-background-clip: text;
    background-clip: text;
}
.metric-card.purple .metric-val {
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    background-clip: text;
}
.metric-card.blue .metric-val {
    background: linear-gradient(135deg, #00d9f5, #3b82f6);
    -webkit-background-clip: text;
    background-clip: text;
}

/* ── Summary Output Box ── */
.summary-box {
    background: rgba(0, 245, 160, 0.05);
    border: 1px solid rgba(0, 245, 160, 0.25);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin: 1rem 0;
    position: relative;
}
.summary-box::before {
    content: '';
    position: absolute;
    top: -1px; left: 20px;
    width: 60px; height: 2px;
    background: linear-gradient(90deg, #00f5a0, #00d9f5);
    border-radius: 2px;
}
.summary-text {
    color: #e2e8f0;
    font-size: 1rem;
    line-height: 1.85;
    font-weight: 400;
}
.summary-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    color: #00f5a0;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 1.5rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-dot {
    width: 6px; height: 6px;
    background: #00f5a0;
    border-radius: 50%;
    display: inline-block;
}

/* ── Text Areas & Inputs ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
}
.stTextArea textarea:focus {
    border-color: rgba(0, 245, 160, 0.4) !important;
    box-shadow: 0 0 0 2px rgba(0, 245, 160, 0.1) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00f5a0, #00d9f5) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(0, 245, 160, 0.3) !important;
}

/* ── Sliders ── */
.stSlider > div > div > div {
    background: linear-gradient(90deg, #00f5a0, #00d9f5) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(10, 10, 20, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #8892a4;
    font-size: 0.88rem;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Info / Warning boxes ── */
.stAlert {
    background: rgba(0, 245, 160, 0.07) !important;
    border: 1px solid rgba(0, 245, 160, 0.2) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

/* ── Comparison table ── */
.compare-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    margin-top: 0.5rem;
}
.compare-table th {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #8892a4;
    padding: 8px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    text-align: left;
}
.compare-table td {
    padding: 10px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: #c9d4e0;
    vertical-align: top;
}
.compare-table tr:hover td {
    background: rgba(255,255,255,0.03);
}
.tag-green { background: rgba(0,245,160,0.15); color: #00f5a0; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; }
.tag-orange { background: rgba(249,115,22,0.15); color: #f97316; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; }
.tag-purple { background: rgba(168,85,247,0.15); color: #a855f7; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8892a4 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0, 245, 160, 0.15) !important;
    color: #00f5a0 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #00f5a0 !important; }

/* ── Copy button ── */
.copy-btn {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    color: #8892a4;
    font-size: 0.75rem;
    padding: 4px 12px;
    border-radius: 8px;
    cursor: pointer;
    margin-top: 8px;
    font-family: 'DM Sans', sans-serif;
}

/* ── Sample pills ── */
.sample-pill {
    display: inline-block;
    background: rgba(168,85,247,0.12);
    border: 1px solid rgba(168,85,247,0.25);
    color: #c084fc;
    font-size: 0.75rem;
    padding: 4px 12px;
    border-radius: 20px;
    cursor: pointer;
    margin: 3px;
    transition: all 0.2s;
}
</style>
""", unsafe_allow_html=True)

# ─── SAMPLE ARTICLES ───────────────────────────────────────────────────────────
SAMPLES = {
    "🏛️ Politics": """Washington (CNN) — The Senate passed a sweeping infrastructure bill Tuesday with rare bipartisan support, allocating $1.2 trillion over eight years to rebuild roads, bridges, and broadband networks across the United States. The legislation, which cleared the chamber with 69 votes, marks one of the largest federal investments in physical infrastructure in decades. President Biden signed the bill in a ceremony at the White House, calling it a once-in-a-generation investment. The package includes $550 billion in new federal spending, covering $110 billion for roads and bridges, $66 billion for passenger and freight rail, $65 billion for broadband expansion, and $55 billion for water infrastructure. Senate Majority Leader Chuck Schumer called the vote a historic moment. Republican Senator Rob Portman, one of the bill's chief architects, said the legislation would help America compete with China. The bill now moves to the House, where progressive Democrats have tied its passage to a larger social spending package. Speaker Pelosi pledged to bring both bills to a vote simultaneously. Critics from the left argued the package does not go far enough on climate provisions, while fiscal conservatives raised concerns about the overall cost.""",

    "🔬 Science": """Scientists at MIT have developed a revolutionary new battery technology that could charge electric vehicles in under five minutes, addressing one of the biggest barriers to widespread EV adoption. The breakthrough involves a new electrode design using carbon nanotubes that dramatically increases the speed at which lithium ions can move through the battery. In laboratory tests, the battery retained 80 percent of its capacity after 10,000 charge cycles, far exceeding the performance of conventional lithium-ion batteries currently used in most electric vehicles. The research team, led by Professor Angela Chen, published their findings in the journal Nature Energy. The technology works by creating a three-dimensional structure within the electrode that provides more surface area for chemical reactions. Unlike conventional batteries that store energy in flat layers, the new design uses a forest of carbon nanotubes that allows ions to enter from multiple directions simultaneously. The researchers estimate that commercial versions of the battery could be available within five to seven years. Several major automakers have already expressed interest in licensing the technology. The batteries also demonstrate improved performance in cold weather, a known weakness of current lithium-ion technology that limits range in winter conditions.""",

    "⚽ Sports": """Barcelona defeated Real Madrid 3-1 in a thrilling El Clasico at Camp Nou on Sunday, with Lionel Messi producing a masterclass performance to seal a crucial victory in the La Liga title race. The Argentine forward scored twice and set up the third goal, tormenting the Madrid defense throughout the match. Real Madrid took a surprise lead in the 23rd minute when Karim Benzema converted from close range after a defensive error. Barcelona responded quickly with Messi equalizing from a trademark left-footed strike. The hosts took the lead just before halftime when Pedri's incisive pass found Antoine Griezmann, who slotted home calmly. Messi completed the scoring with a penalty in the 71st minute after Luis Suarez was brought down in the box. The result leaves Barcelona three points clear at the top of the table with eight games remaining. Manager Ronald Koeman praised his team's resilience and attacking play after the match. Real Madrid manager Zinedine Zidane was philosophical in defeat, insisting the title race is not over. The result means Barcelona need just four wins from their remaining games to secure the championship."""
}

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800;
             background: linear-gradient(90deg,#00f5a0,#a855f7);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;
             background-clip:text;">⚡ SummarAI</div>
        <div style="color:#8892a4; font-size:0.75rem; margin-top:4px;">TF-IDF + Cosine Similarity</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    top_n = st.slider("Sentences in summary", min_value=1, max_value=7, value=3,
                      help="Number of top sentences to extract")

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    <div style="font-size:0.82rem; color:#8892a4; line-height:1.8;">
    <b style="color:#00f5a0;">Method:</b> Extractive<br>
    <b style="color:#00f5a0;">Vectorizer:</b> TF-IDF<br>
    <b style="color:#00f5a0;">Similarity:</b> Cosine<br>
    <b style="color:#00f5a0;">N-grams:</b> (1, 2)<br>
    <b style="color:#00f5a0;">Max features:</b> 10,000<br>
    <b style="color:#00f5a0;">Position bias:</b> ✅ Enabled<br>
    <b style="color:#00f5a0;">Dataset:</b> CNN/DailyMail<br>
    <b style="color:#00f5a0;">Train size:</b> 284K rows
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏆 ROUGE Scores")
    rouge_data = pd.DataFrame({
        'Metric': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
        'Val': ['—', '—', '—'],
        'Test': ['0.3254+', '0.1231+', '0.2106+']
    })
    st.dataframe(rouge_data, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#8892a4; text-align:center; line-height:1.6;">
    Built with ❤️ using<br>Streamlit · scikit-learn · Plotly<br>
    CNN/DailyMail Dataset
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN CONTENT ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">⚡ NLP · Extractive Summarization</div>
    <div class="hero-title">SummarAI</div>
    <div class="hero-sub">TF-IDF + Cosine Similarity · CNN/DailyMail · 284K Articles</div>
</div>
<div class="glow-divider"></div>
""", unsafe_allow_html=True)

# ─── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["⚡ Summarize", "📊 Analysis", "🔬 Compare Methods"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — SUMMARIZE
# ══════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header"><span class="section-dot"></span> Input Article</div>', unsafe_allow_html=True)

        # Sample pills
        st.markdown("**Try a sample:**")
        sample_cols = st.columns(3)
        selected_sample = None
        for i, (label, _) in enumerate(SAMPLES.items()):
            with sample_cols[i]:
                if st.button(label, key=f"sample_{i}"):
                    selected_sample = label

        article_input = st.text_area(
            label="Paste your article here",
            value=SAMPLES[selected_sample] if selected_sample else "",
            height=320,
            placeholder="Paste any news article, blog post, or long-form text here...",
            label_visibility="collapsed"
        )

        word_count = len(article_input.split()) if article_input.strip() else 0
        st.markdown(f'<div style="font-size:0.8rem; color:#8892a4; margin-top:4px;">📝 {word_count} words · {len(article_input)} characters</div>', unsafe_allow_html=True)

        summarize_btn = st.button("⚡ Generate Summary", type="primary")

    with col_right:
        st.markdown('<div class="section-header"><span class="section-dot"></span> Generated Summary</div>', unsafe_allow_html=True)

        if summarize_btn and article_input.strip():
            with st.spinner("Analyzing sentences..."):
                time.sleep(0.4)
                summary = cosine_tfidf_summary(article_input, top_n=top_n)
                stats   = compression_stats(article_input, summary)

            # Metrics
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-val">{stats['article_words']}</div>
                    <div class="metric-label">Input words</div>
                </div>
                <div class="metric-card blue">
                    <div class="metric-val">{stats['summary_words']}</div>
                    <div class="metric-label">Output words</div>
                </div>
                <div class="metric-card orange">
                    <div class="metric-val">{stats['compression_pct']}%</div>
                    <div class="metric-label">Kept</div>
                </div>
                <div class="metric-card purple">
                    <div class="metric-val">{stats['summary_sentences']}</div>
                    <div class="metric-label">Sentences</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Summary output
            st.markdown(f"""
            <div class="summary-box">
                <div class="summary-label">✦ Summary</div>
                <div class="summary-text">{summary}</div>
            </div>
            """, unsafe_allow_html=True)

            # Store in session
            st.session_state['last_article'] = article_input
            st.session_state['last_summary'] = summary
            st.session_state['last_stats']   = stats

            # Copy button (workaround)
            st.code(summary, language=None)

        elif summarize_btn:
            st.warning("Please paste an article first.")
        else:
            st.markdown("""
            <div style="text-align:center; padding: 4rem 1rem; color:#8892a4;">
                <div style="font-size:3rem; margin-bottom:1rem;">⚡</div>
                <div style="font-family:'Syne',sans-serif; font-size:1rem; color:#e2e8f0; margin-bottom:0.5rem;">Ready to summarize</div>
                <div style="font-size:0.85rem;">Paste an article and click Generate</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header"><span class="section-dot"></span> Sentence Score Analysis</div>', unsafe_allow_html=True)

    article_for_analysis = st.session_state.get('last_article', '')

    if article_for_analysis:
        sentences, scores = get_sentence_scores(article_for_analysis)

        if sentences and scores:
            top_n_val = st.session_state.get('top_n_used', top_n)
            top_indices = sorted(
                sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
            )

            # Bar chart of scores
            colors = ['#00f5a0' if i in top_indices else '#334155'
                      for i in range(len(sentences))]

            fig = go.Figure(go.Bar(
                x=[f"S{i+1}" for i in range(len(sentences))],
                y=scores,
                marker_color=colors,
                hovertemplate='<b>Sentence %{x}</b><br>Score: %{y:.4f}<br><extra></extra>',
            ))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8892a4', family='DM Sans'),
                xaxis=dict(showgrid=False, color='#8892a4'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#8892a4'),
                margin=dict(l=10, r=10, t=30, b=10),
                height=280,
                title=dict(text="Cosine similarity score per sentence  (green = selected)", font=dict(size=13, color='#8892a4')),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Sentence table
            st.markdown('<div class="section-header"><span class="section-dot"></span> All Sentences Ranked</div>', unsafe_allow_html=True)

            ranked = sorted(enumerate(zip(sentences, scores)), key=lambda x: x[1][1], reverse=True)
            rows = ""
            for rank, (orig_idx, (sent, score)) in enumerate(ranked):
                selected = "✅" if orig_idx in top_indices else ""
                bar_w = int(score / max(scores) * 100)
                rows += f"""
                <tr>
                    <td style="color:#8892a4;">#{rank+1}</td>
                    <td style="font-size:0.82rem; max-width:380px;">{sent[:120]}{'...' if len(sent)>120 else ''}</td>
                    <td>
                        <div style="display:flex; align-items:center; gap:8px;">
                            <div style="background:rgba(255,255,255,0.06); border-radius:4px; height:6px; width:80px; overflow:hidden;">
                                <div style="width:{bar_w}%; height:100%; background:linear-gradient(90deg,#00f5a0,#00d9f5); border-radius:4px;"></div>
                            </div>
                            <span style="font-size:0.8rem; color:#00f5a0;">{score:.4f}</span>
                        </div>
                    </td>
                    <td style="text-align:center;">{selected}</td>
                </tr>"""

            st.markdown(f"""
            <table class="compare-table">
                <thead><tr>
                    <th>Rank</th><th>Sentence</th><th>Score</th><th>Selected</th>
                </tr></thead>
                <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)
    else:
        st.info("Generate a summary first in the Summarize tab to see the analysis.")

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARE METHODS
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header"><span class="section-dot"></span> Method Comparison</div>', unsafe_allow_html=True)

    # ROUGE comparison chart
    methods = ['TF-IDF + Cosine<br>(Our Model)', 'RF/XGBoost<br>(Compression Ratio)', 'Random Baseline', 'BERT (Reference)']
    rouge1  = [0.3254, 0.0, 0.15, 0.48]
    rouge2  = [0.1231, 0.0, 0.04, 0.23]
    rougeL  = [0.2106, 0.0, 0.13, 0.45]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name='ROUGE-1', x=methods, y=rouge1,
                          marker_color='#00f5a0', opacity=0.9))
    fig2.add_trace(go.Bar(name='ROUGE-2', x=methods, y=rouge2,
                          marker_color='#a855f7', opacity=0.9))
    fig2.add_trace(go.Bar(name='ROUGE-L', x=methods, y=rougeL,
                          marker_color='#f97316', opacity=0.9))
    fig2.update_layout(
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8892a4', family='DM Sans'),
        xaxis=dict(showgrid=False, color='#8892a4'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                   color='#8892a4', title='F1 Score'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8892a4')),
        margin=dict(l=10, r=10, t=30, b=10),
        height=320,
        title=dict(text="ROUGE scores across methods", font=dict(size=13, color='#8892a4')),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Method comparison table
    st.markdown('<div class="section-header"><span class="section-dot"></span> Method Details</div>', unsafe_allow_html=True)

    st.markdown("""
    <table class="compare-table">
        <thead><tr>
            <th>Method</th>
            <th>Type</th>
            <th>Target</th>
            <th>Evaluation</th>
            <th>ROUGE-1</th>
            <th>Verdict</th>
        </tr></thead>
        <tbody>
            <tr>
                <td><b style="color:#e2e8f0;">TF-IDF + Cosine</b></td>
                <td><span class="tag-green">Extractive</span></td>
                <td>Top sentences</td>
                <td>ROUGE F1</td>
                <td style="color:#00f5a0; font-weight:600;">0.3254</td>
                <td><span class="tag-green">✅ Best</span></td>
            </tr>
            <tr>
                <td><b style="color:#e2e8f0;">Random Forest</b></td>
                <td><span class="tag-orange">Regression</span></td>
                <td>Compression ratio</td>
                <td>R², RMSE</td>
                <td style="color:#8892a4;">—</td>
                <td><span class="tag-orange">⚠️ Baseline</span></td>
            </tr>
            <tr>
                <td><b style="color:#e2e8f0;">XGBoost</b></td>
                <td><span class="tag-orange">Regression</span></td>
                <td>Compression ratio</td>
                <td>R²: 0.638</td>
                <td style="color:#8892a4;">—</td>
                <td><span class="tag-orange">⚠️ Baseline</span></td>
            </tr>
            <tr>
                <td><b style="color:#e2e8f0;">Cosine + RF/XGB</b></td>
                <td><span class="tag-purple">Hybrid</span></td>
                <td>Sentence rank</td>
                <td>ROUGE F1</td>
                <td style="color:#c084fc;">~0.33</td>
                <td><span class="tag-purple">🔬 Combined</span></td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Key insights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card" style="text-align:left; padding:1.2rem;">
            <div style="font-size:1.5rem; margin-bottom:6px;">🎯</div>
            <div style="font-family:'Syne',sans-serif; font-size:0.85rem; font-weight:700; color:#e2e8f0; margin-bottom:6px;">Why TF-IDF wins</div>
            <div style="font-size:0.8rem; color:#8892a4; line-height:1.6;">Directly scores sentence relevance to the full article using cosine angle. No training needed — unsupervised and fast.</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align:left; padding:1.2rem;">
            <div style="font-size:1.5rem; margin-bottom:6px;">📐</div>
            <div style="font-family:'Syne',sans-serif; font-size:0.85rem; font-weight:700; color:#e2e8f0; margin-bottom:6px;">RF/XGB role</div>
            <div style="font-size:0.8rem; color:#8892a4; line-height:1.6;">Predicts compression ratio as a numeric regression task. R²=0.638 means 63% variance explained — useful as a document-length predictor.</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card" style="text-align:left; padding:1.2rem;">
            <div style="font-size:1.5rem; margin-bottom:6px;">🚀</div>
            <div style="font-family:'Syne',sans-serif; font-size:0.85rem; font-weight:700; color:#e2e8f0; margin-bottom:6px;">Next step</div>
            <div style="font-size:0.8rem; color:#8892a4; line-height:1.6;">BART or T5 transformer fine-tuned on CNN/DailyMail would push ROUGE-1 above 0.44 via abstractive generation.</div>
        </div>""", unsafe_allow_html=True)
