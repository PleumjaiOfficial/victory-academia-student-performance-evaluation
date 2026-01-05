import time
import calendar
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import sys 
import random
import os
from os import environ
from dotenv import load_dotenv
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
load_dotenv()

try:
    parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    sys.path.append(str(parent_path))

    current_path = Path(__file__).parent
    sys.path.append(str(current_path))
    
    from common.algorithms.ranking import toptalent as ranking
    from common.algorithms.som import SOM
    
    print("✅ Modules loaded successfully")
except ImportError as e:
    st.error(f"❌ unsuccess loaded: {e}")
    st.stop()

warnings.filterwarnings("ignore")
np.random.seed(42)

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Student Growth Profile",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# Initialize Session State
# =========================
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False
if 'selected_level' not in st.session_state:
    st.session_state.selected_level = ""
if 'selected_month' not in st.session_state:
    st.session_state.selected_month = ""
if 'selected_alias' not in st.session_state:
    st.session_state.selected_alias = ""


# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_data(level, month):
    current_dir = Path(__file__).parent
    file_path = current_dir / "mock_data" / "student_performance_prepared_2025.xlsx"
    df = pd.read_excel(file_path, sheet_name=level)
    return df.query(f"MONTH == '{month}'")

# =========================
# 1.SIMULATION
# =========================
def generate_sim_students(
    base_df,
    SUBJECTS,
    n_sim_students=100,
    score_profile=None,
    min_score=0,
    max_score=100,
    random_seed=None
):
    """
    Keep original students + add simulated students for analysis
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    if score_profile is None:
        score_profile = {
            sub: {
                'mean': base_df[sub].mean(),
                'std': base_df[sub].std(ddof=0)
            }
            for sub in SUBJECTS
        }

    sim_rows = []

    for i in range(1, n_sim_students + 1):
        row = {
            'MONTH': base_df['MONTH'].iloc[0],
            'LEVELS': base_df['LEVELS'].iloc[0],
            'ALIAS': f'SIM_{i}'
        }

        for sub in SUBJECTS:
            mean = score_profile[sub]['mean']
            std = score_profile[sub]['std']

            score = np.random.normal(mean, std)
            score = np.clip(score, min_score, max_score)

            row[sub] = round(score, 2)

        # topic คงเดิม
        row['MATH_TOPIC'] = base_df['MATH_TOPIC'].iloc[0]
        row['SCIENCE_TOPIC'] = base_df['SCIENCE_TOPIC'].iloc[0]
        row['ENGLISH_TOPIC'] = base_df['ENGLISH_TOPIC'].iloc[0]
        if 'THAI_TOPIC' in base_df.columns:
            row['THAI_TOPIC'] = base_df['THAI_TOPIC'].iloc[0]

        sim_rows.append(row)

    sim_df = pd.DataFrame(sim_rows)
    combined_df = pd.concat([base_df, sim_df], ignore_index=True)

    return combined_df


# =========================
# 2.FEATURE ENGINEERING
# =========================
def prepare_data_for_zoing_analysis(df, SUBJECTS):
    df["STEM_AVG"] = (df["MATH"] + df["SCIENCE"]) / 2
    
    if "THAI" in SUBJECTS:
        df["LANG_AVG"] = (df["ENGLISH"] + df["THAI"]) / 2
    else:
        df["LANG_AVG"] = df["ENGLISH"]
    
    df["OVERALL_AVG"] = df[SUBJECTS].mean(axis=1)
    return df

# =========================
# 3.RANKING
# =========================
def ranking_students(df, SUBJECTS):
    ranking_df = df.rename(columns={'ALIAS': 'PERSON_ID'})

    print("Config for ranking")
    num_scores = len(SUBJECTS) 
    weights = [100 / num_scores] * num_scores
    bool_flags = [True] * num_scores
    print("evaluate_subject: ", SUBJECTS)
    print("weights: ", weights)
    print("bool_flags: ", bool_flags)

    # Ranking Analysis
    toptalent, tier_toptalent = ranking(ranking_df, SUBJECTS, weights, bool_flags)
    ranking_df = ranking_df.merge(toptalent, on="PERSON_ID")
    ranking_df["TALENT_SCORE"] = (ranking_df["TALENT_SCORE"] * 100).round(2)
    ranking_df = ranking_df.sort_values(by=["TALENT_SCORE"], ascending=[False])
    ranking_df["RANK"] = ranking_df["TALENT_SCORE"].rank(ascending=False, method='min').astype(int)
    ranking_df = ranking_df.drop(columns=['TIER'])

    return ranking_df


# =========================
# CLUSTERING 
# =========================
def cluster_students(df, SUBJECTS):
    df = df.copy(deep=True)
    cluster_features = df[SUBJECTS].values
    cluster, centroids, xy = SOM(cluster_features, K=3, random_state=42, show=False, return_xy=True)
    df["CLUSTER"] = cluster

    return df


# =========================
# FULL PIPELINE
# =========================
def process_pipeline(student_df, score_profile, SUBJECTS, n_sim_students=40):
    sim_df = generate_sim_students(student_df, SUBJECTS, n_sim_students=n_sim_students, score_profile=score_profile)
    feature_df = prepare_data_for_zoing_analysis(sim_df, SUBJECTS)
    ranked_df = ranking_students(feature_df, SUBJECTS)
    cluster_df = cluster_students(ranked_df, SUBJECTS)

    results_df = cluster_df.copy(deep=True)
    results_df = results_df.sort_values(by=["TALENT_SCORE"], ascending=[False])
    results_df = results_df.reset_index(drop=True)

    return results_df




# =========================
# VISUALIZATION FUNCTIONS
# =========================
def plot_learning_mode_bar(
    student_df,
    title="📊 จำนวนครั้งการเรียนแยกตามรูปแบบ"
):
    
    learning_modes = [
        "บรรยายและทำแบบฝึกหัด",
        "ทำแบบฝึกหัดระดับกลาง",
        "ทดสอบ",
        "ทดลองทางวิทยาศาสตร์"
    ]

    values = [
        student_df.iloc[0].get(mode, 0)
        for mode in learning_modes
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=learning_modes,
                y=values,
                text=values,
                textposition="auto"
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="รูปแบบการเรียน",
        yaxis_title="จำนวนครั้ง",
        height=400,
    )

    fig.update_layout(
        hovermode=False,
        dragmode=False
    )

    return fig

def plot_student_score_radar(
    student_df,
    level,
    title="📊 Student Performance Radar"
):
    """
    สร้าง Radar Chart คะแนนรายวิชา
    """

    # กำหนดวิชาตามระดับชั้น
    if level.lower().startswith("primary"):
        subjects = ["MATH", "SCIENCE", "THAI", "ENGLISH"]
    else:
        subjects = ["MATH", "SCIENCE", "ENGLISH"]

    # ดึงคะแนน (ถ้าไม่มี column ให้เป็น 0)
    scores = [student_df.iloc[0].get(subj, 0) for subj in subjects]

    # Radar ต้องปิด loop
    subjects_radar = subjects + [subjects[0]]
    scores_radar = scores + [scores[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=scores_radar,
            theta=subjects_radar,
            fill="toself",
            name="Score"
        )
    )

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        height=500
    )

    fig.update_layout(
        hovermode=False,
        dragmode=False
    )
    
    config = {
        'displayModeBar': False,  # ซ่อน toolbar
        'staticPlot': True  # ทำให้กราฟเป็น static
    }
    
    topic_lines = []
    for subj in subjects:
        topic_col = f"{subj}_TOPIC"
        topic = student_df.iloc[0].get(topic_col, None)
        if pd.notna(topic):
            topic_lines.append(f"• {subj}: {topic}")

    topic_text = "<br>".join(topic_lines) if topic_lines else "ไม่มีข้อมูลหัวข้อที่ทดสอบ"

    return fig, topic_text

def plot_results_target_with_simulation(
    df,
    target,
    x_col='STEM_AVG',
    y_col='LANG_AVG',
    cluster_col='CLUSTER',
    label_col='PERSON_ID',
    title='Student Performance vs Simulation'
):
    """
    Plot zoning analysis with simulation
    """
    # Filter: target + SIM only
    plot_df = df[
        (df[label_col] == target) |
        (df[label_col].str.startswith("SIM"))
    ].copy()

    fig = go.Figure()

    # Zones
    zones = [
        (0, 50, 0, 50, "Warning Zone", "#f9ebea"),
        (0, 50, 50, 80, "STEM Support", "#fef9e7"),
        (0, 50, 80, 100, "Language Expert", "#eafaf1"),
        (50, 80, 0, 50, "Language Support", "#fef5e7"),
        (50, 80, 50, 80, "Development Zone", "#e8f8f5"),
        (80, 100, 0, 50, "STEM Expert", "#f4ecf7"),
        (80, 100, 50, 80, "STEM Strong", "#e8daef"),
        (50, 80, 80, 100, "Language Strong", "#eaf2f8"),
        (80, 100, 80, 100, "Perfect Zone", "#d4efdf")
    ]

    for xmin, xmax, ymin, ymax, label, color in zones:
        fig.add_shape(
            type="rect",
            x0=xmin, x1=xmax,
            y0=ymin, y1=ymax,
            fillcolor=color,
            opacity=0.35,
            line_width=0,
            layer="below"
        )
        fig.add_annotation(
            x=(xmin + xmax) / 2,
            y=(ymin + ymax) / 2,
            text=label,
            showarrow=False,
            font=dict(size=10)
        )

    # Plot Simulation
    sim_df = plot_df[plot_df[label_col].str.startswith("SIM")]

    fig.add_trace(go.Scatter(
        x=sim_df[x_col],
        y=sim_df[y_col],
        mode='markers',
        marker=dict(
            size=10,
            color='gray',
            symbol='circle',
            opacity=0.5
        ),
        hoverinfo='skip',
        name='Simulation'
    ))

    # Plot Target
    target_df = plot_df[plot_df[label_col] == target]

    fig.add_trace(go.Scatter(
        x=target_df[x_col],
        y=target_df[y_col],
        mode='markers+text',
        marker=dict(
            size=14,
            color='red',
            symbol='star',
            line=dict(width=1, color='black')
        ),
        text=[target],
        textposition='top center',
        textfont=dict(size=12),
        name='Your Child'
    ))

    # Reference Lines
    for v in [50, 80]:
        fig.add_shape(type="line", x0=v, x1=v, y0=0, y1=100,
                      line=dict(color="red" if v == 50 else "gray", dash="dash"))
        fig.add_shape(type="line", x0=0, x1=100, y0=v, y1=v,
                      line=dict(color="red" if v == 50 else "gray", dash="dash"))

    # Layout
    fig.update_layout(
        title=title,
        xaxis=dict(title='STEM Average', range=[0, 100]),
        yaxis=dict(title='Language Average', range=[0, 100]),
        plot_bgcolor='white',
        template='plotly_white',
        height=700,
        showlegend=True,
        hovermode=False,
        dragmode=False
    )

    return fig


def plot_ranking_comparison(df, target, top_n=20):
    """
    แสดงอันดับของนักเรียนเทียบกับนักเรียนจำลอง
    """
    # เลือกเฉพาะ Top N และนักเรียนที่สนใจ
    top_df = df.nsmallest(top_n, 'RANK').copy()
    
    # ถ้านักเรียนไม่อยู่ใน Top N ให้เพิ่มเข้าไป
    if target not in top_df['PERSON_ID'].values:
        target_row = df[df['PERSON_ID'] == target]
        top_df = pd.concat([top_df, target_row])
    
    top_df = top_df.sort_values('RANK')
    
    # สร้างสีแยกระหว่างนักเรียนจริงกับจำลอง
    colors = ['red' if pid == target else 'lightblue' if pid.startswith('SIM') else 'blue' 
              for pid in top_df['PERSON_ID']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_df['PERSON_ID'],
            y=top_df['TALENT_SCORE'],
            marker_color=colors,
            text=top_df['RANK'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Score: %{y:.2f}<br>Rank: %{text}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f'🏆 Top {top_n} Rankings (ลูกของคุณเป็นสีแดง)',
        xaxis_title='Student ID',
        yaxis_title='Talent Score',
        height=500,
        showlegend=False,
        xaxis={'tickangle': -45}
    )
    
    return fig


# =========================
# MAIN APP
# =========================
def main():

    LEVELS = ["Primary6"]
    MONTHS = ["Dec"]

    logo_path = Path(__file__).parent / "bd-logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=2000)

    st.title("Bewdar Academy Lamphun: Student Growth Profile 📊")

    st.markdown("""
    ### 🎯 ระบบวิเคราะห์และพยากรณ์ผลการเรียนของนักเรียน
    
    ระบบนี้จะช่วยคุณวิเคราะห์:
    - 📈 **สถิติการเรียน**: จำนวนครั้งที่เข้าเรียน, เวลาเรียน, และรูปแบบการเรียน
    - 🎯 **ผลการทดสอบ**: คะแนนในแต่ละวิชา พร้อมแสดงจุดแข็ง-จุดอ่อน
    - 🔮 **การจำลองห้องเรียน**: พยากรณ์อันดับของลูกคุณเมื่อเทียบกับเพื่อนในห้อง
    - 🗺️ **การวิเคราะห์โซน**: ระบุตำแหน่งของนักเรียนและแนะนำแผนพัฒนาที่เหมาะสม
    
    ---
    """)

    # ======================
    # Sidebar
    # ======================
    with st.sidebar:
        st.header("🧒 Student Information")

        selected_level = st.selectbox(
            "🏫 ระดับชั้น", 
            [""] + LEVELS,
            key='level_select'
        )
        selected_month = st.selectbox(
            "📅 เดือน", 
            [""] + MONTHS,
            key='month_select'
        )

        selected_alias = None
        df_tmp = None

        if selected_level and selected_month:
            df_tmp = load_data(selected_level, selected_month)
            alias_list = sorted(df_tmp["ALIAS"].unique())
            alias_list = [alias for alias in alias_list if alias != "Full-Score"]
            selected_alias = st.selectbox(
                "👤 เลือกนักเรียน", 
                [""] + alias_list,
                key='alias_select'
            )

        # ใช้ button เพื่อเริ่มการวิเคราะห์
        if st.button("🚀 เริ่มการวิเคราะห์", type="primary"):
            if selected_level and selected_month and selected_alias:
                st.session_state.analysis_started = True
                st.session_state.selected_level = selected_level
                st.session_state.selected_month = selected_month
                st.session_state.selected_alias = selected_alias
            else:
                st.warning("⚠️ กรุณาเลือกข้อมูลนักเรียนให้ครบก่อน")

    # ======================
    # Main Content
    # ======================
    if st.session_state.analysis_started:
        # โหลดข้อมูลจาก session state
        selected_level = st.session_state.selected_level
        selected_month = st.session_state.selected_month
        selected_alias = st.session_state.selected_alias
        
        # กำหนด SUBJECTS ตามระดับชั้น
        if 'Primary' in selected_level:
            SUBJECTS = ["MATH", "SCIENCE", "ENGLISH", "THAI"]
        else:
            SUBJECTS = ["MATH", "SCIENCE", "ENGLISH"]

        df_tmp = load_data(selected_level, selected_month)
        student_df = df_tmp[df_tmp["ALIAS"] == selected_alias]
        
        st.subheader(f"📊 สถิติโดยรวมของ: {selected_alias}")
        st.dataframe(student_df)
        learning_duration = student_df.iloc[0]["TOTAL_CLASSES"]
        learning_duration_time = learning_duration * 60
        attended_days = (student_df.iloc[0]["TOTAL_CLASSES"]/student_df.iloc[0]["TOTAL_DAYS"])*100

        # แบบจำลอง performance
        col1, col2, col3 = st.columns(3)
        col1.metric("จำนวนครั้งที่เข้าเรียน (ครั้ง)", student_df.iloc[0]["TOTAL_CLASSES"] , help="เริ่มเรียนตั้งแต่วันที่ " + str(student_df.iloc[0]['START_RECORD'].date()))
        col2.metric("เวลาเรียนโดยเฉลี่ย (นาที)", learning_duration_time)
        col3.metric("เข้าเรียนสำเร็จ (%)", round(attended_days,2) , help = f"จำนวนวันทั้งหมด {student_df.iloc[0]['TOTAL_DAYS']}")

        # bar plot แสดงการเรียนในแต่ละโหมด
        fig = plot_learning_mode_bar(student_df)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader(f"🚀 ผลลัพธ์ของการทดสอบของ: {selected_alias}")

        fig, topic_text = plot_student_score_radar(
            student_df=student_df,
            level=selected_level,
            title=f"📊 ผลการเรียนของ {selected_alias}"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## 📝 หัวข้อที่มีการทดสอบ")
        st.markdown(topic_text, unsafe_allow_html=True)

        st.divider()

        # ======================
        # Simulation Section
        # ======================
        st.subheader(f"🔮 การจำลองห้องเรียนและพยากรณ์อันดับของ: {selected_alias}")
        st.markdown("ปรับแต่งพารามิเตอร์ของห้องเรียนจำลองเพื่อดูว่าลูกของคุณจะอยู่อันดับไหนในห้องเรียน")

        # Simulation Parameters
        st.markdown("### ⚙️ ตั้งค่าห้องเรียนจำลอง")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_sim_students = st.slider(
                "จำนวนนักเรียนจำลอง",
                min_value=10,
                max_value=100,
                value=40,
                step=5,
                help="จำนวนนักเรียนที่จะสร้างขึ้นมาเพื่อเปรียบเทียบ",
                key='n_sim'
            )
        
        with col2:
            overall_mean = st.slider(
                "คะแนนเฉลี่ยโดยรวมของห้อง",
                min_value=30.0,
                max_value=90.0,
                value=65.0,
                step=5.0,
                help="คะแนนเฉลี่ยของนักเรียนในห้องเรียนจำลอง",
                key='overall_mean'
            )
        
        st.markdown("### 📊 ปรับคะแนนเฉลี่ยและความกระจายของแต่ละวิชา")
        
        # สร้าง profile แยกตามวิชา
        score_profile = {}
        
        # แบ่ง columns ตามจำนวนวิชา
        n_cols = len(SUBJECTS)
        cols = st.columns(n_cols)
        
        for idx, subject in enumerate(SUBJECTS):
            with cols[idx]:
                st.markdown(f"**{subject}**")
                mean = st.number_input(
                    f"ค่าเฉลี่ย",
                    min_value=0.0,
                    max_value=100.0,
                    value=overall_mean,
                    step=5.0,
                    key=f"mean_{subject}"
                )
                std = st.number_input(
                    f"ความกระจาย (SD)",
                    min_value=5.0,
                    max_value=30.0,
                    value=20.0,
                    step=5.0,
                    key=f"std_{subject}",
                    help="ความกระจายของคะแนน (ยิ่งมากยิ่งกระจาย)"
                )
                score_profile[subject] = {'mean': mean, 'std': std}

        st.divider()

        # Run Simulation - ทำงานทันทีเมื่อมีการเปลี่ยนแปลง
        with st.spinner('กำลังจำลองข้อมูลและวิเคราะห์...'):
            result_df = process_pipeline(student_df, score_profile, SUBJECTS, n_sim_students=n_sim_students)
            
            # หาข้อมูลของนักเรียน
            target_data = result_df[result_df['PERSON_ID'] == selected_alias].iloc[0]
            target_rank = target_data['RANK']
            target_score = target_data['TALENT_SCORE']
            total_students = len(result_df)

        # แสดงผลลัพธ์
        st.success("✅ วิเคราะห์เสร็จสมบูรณ์")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("อันดับของลูกคุณ", f"{target_rank} / {total_students}")
        col2.metric("คะแนนความสามารถ", f"{target_score:.2f}")
        percentile = ((total_students - target_rank + 1) / total_students) * 100
        col3.metric("เปอร์เซ็นไทล์", f"{percentile:.1f}%")
        col4.metric("นักเรียนที่ดีกว่า", f"{target_rank - 1} คน")

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📍 Zoning Analysis", "🏆 Ranking Comparison", "📊 Full Data"])
        
        with tab1:
            st.markdown("### 📍 ตำแหน่งของลูกคุณในห้องเรียนจำลอง")
            fig_zone = plot_results_target_with_simulation(
                result_df, 
                target=selected_alias,
                title=f'การวิเคราะห์โซนของ {selected_alias} (ดาวสีแดง) เทียบกับนักเรียนจำลอง'
            )
            st.plotly_chart(fig_zone, use_container_width=True)
            
            # คำอธิบาย Zones
            with st.expander("📖 คำอธิบายโซนต่างๆ"):
                st.markdown("""**การวิเคราะห์นี้แสดงให้เห็นว่า นักเรียนมีจุดแข็งและจุดอ่อนอย่างไร และเราจะพัฒนานักเรียนอย่างไร**
                            
                        โดยพื้นที่จะแบ่งออกเป็น 9 พื้นที่ ได้แก่:
                        1. Warning Zone (พื้นที่เตือนภัย)
                            - สถานการณ์ปัจจุบัน: นักเรียนยังไม่เข้าใจพื้นฐาน ต้องได้รับการสนับสนุนจากโรงเรียนและผู้ปกครอง
                            - แผนพัฒนาขั้นต่อไป: ปรับหลักสูตรใหม่ให้เหมาะสม, ร่วมมือกับผู้ปกครองอย่างใกล้ชิด และจัดการเรียนแบบกลุ่มเล็กแยกออกมา
                            - เป้าหมายต่อไป: เพื่อให้นักเรียนเข้าออกจากพื้นนี้ไปเข้าสู่ พื้นที่พัฒนาการ (Development Zone)

                        2. STEM Support (พื้นที่สนับสนุน)
                            - สถานการณ์ปัจจุบัน: นักเรียนมีพื้นฐานด้านภาษา แต่ต้องเสริมด้านวิทยาศาสตร์-คณิต
                            - แผนพัฒนาขั้นต่อไป: ให้คุณครูสาย STEM ดูแลอย่างใกล้ชิด, เพิ่มกิจกรรมการทดลองเข้ามาเพื่อทำให้เข้าใจง่าย และจัดการเรียนแบบกลุ่มเล็กแยกออกมา
                            - เป้าหมายต่อไป: เพื่อให้นักเรียนเข้าออกจากพื้นนี้ไปเข้าสู่ พื้นที่พัฒนาการ (Development Zone)

                        3. Language Support (พื้นที่สนับสนุน)
                            - สถานการณ์ปัจจุบัน: นักเรียนมีพื้นฐานด้านคณิต-วิทย์ แต่ต้องเสริมด้านภาษา
                            - แผนพัฒนาขั้นต่อไป: ให้คุณครูสายภาษาดูแลอย่างใกล้ชิด, เพิ่มแบบฝึกหัดให้ทำอย่างสม่ำเสมอ และจัดการเรียนแบบกลุ่มเล็กแยกออกมา
                            - เป้าหมายต่อไป: เพื่อให้นักเรียนเข้าออกจากพื้นนี้ไปเข้าสู่ พื้นที่พัฒนาการ (Development Zone)

                        4. Development Zone (พื้นที่พัฒนาการ)
                            - สถานการณ์ปัจจุบัน: นักเรียนมีพื้นฐานดีทุกวิชาแล้ว แต่ยังไม่มีจุดเด่นที่ชัดเจน
                            - แผนพัฒนาขั้นต่อไป: จัดกลุ่มการเรียนแบบผสมผสานให้เรียนกับเพื่อนที่เก่งหลายๆ แบบ และทดสอบนักเรียนภายใต้ทักษะหลายๆ อย่าง
                            - เป้าหมายต่อไป: เพื่อให้นักเรียนเข้าออกจากพื้นนี้ไปเข้าสู่ พื้นที่โดดเด่น(Strong Zone) หรือ พื้นที่สมสมบูรณ์แบบ(Perfect Zone)
                        
                        5. Language Expert (พื้นที่พัฒนาการ)
                            - สถานการณ์ปัจจุบัน: นักเรียนมีพื้นฐานภาษาดีมาก ต้องเสริมการคิดวิเคราะห์
                            - แผนพัฒนาขั้นต่อไป: จัดกลุ่มการเรียนให้เรียนกับเพื่อนที่เก่งด้านการวิเคราะห์และวิทยาศาสตร์ และทดสอบด้านการวิเคราะห์และแก้ปัญหาอย่างสม่ำเสมอ
                            - เป้าหมายต่อไป: เพื่อให้นักเรียนเข้าออกจากพื้นนี้ไปเข้าสู่ พื้นที่โดดเด่น(Strong Zone) หรือ พื้นที่สมสมบูรณ์แบบ(Perfect Zone)
                            
                        6. STEM Expert (พื้นที่พัฒนาการ)
                            - สถานการณ์ปัจจุบัน: นักเรียนมีพื้นฐานด้านการวิเคราะห์และคณิตศาสตร์ดีมาก ต้องเสริมด้านภาษา
                            - แผนพัฒนาขั้นต่อไป: จัดกลุ่มการเรียนให้เรียนกับเพื่อนที่เก่งด้านภาษา และทดสอบด้านภาษาอย่างสม่ำเสมอ
                            - เป้าหมายต่อไป: เพื่อให้นักเรียนเข้าออกจากพื้นนี้ไปเข้าสู่ พื้นที่โดดเด่น(Strong Zone) หรือ พื้นที่สมสมบูรณ์แบบ(Perfect Zone)
                            
                        7. STEM Strong (พื้นที่โดดเด่น) 
                            - สถานการณ์ปัจจุบัน: นักเรียนเชี่ยวชาญด้านการวิเคราะห์และวิทยาศาสตร์ และมีพื้นฐานภาษาที่ดีในเวลาเดียวกัน
                            - แผนพัฒนาขั้นต่อไป: จัดกลุ่มการเรียนให้เรียนกับเพื่อนที่เก่งด้านภาษา และพยายามประคับประคองให้รักษามาตรฐานต่อไป
                            - เป้าหมายต่อไป: เพื่อให้นักเรียนเข้าออกจากพื้นนี้ไปเข้าสู่ พื้นที่สมสมบูรณ์แบบ(Perfect Zone)

                        8. Language Strong (พื้นที่โดดเด่น) 
                            - สถานการณ์ปัจจุบัน: นักเรียนเชี่ยวชาญด้านภาษา และมีพื้นฐานการวิเคราะห์และวิทยาศาสตร์ที่ดีในเวลาเดียวกัน
                            - แผนพัฒนาขั้นต่อไป: จัดกลุ่มการเรียนให้เรียนกับเพื่อนที่เก่งด้านการวิเคราะห์และวิทยาศาสตร์ และพยายามประคับประคองให้รักษามาตรฐานต่อไป
                            - เป้าหมายต่อไป: เพื่อให้นักเรียนเข้าออกจากพื้นนี้ไปเข้าสู่ พื้นที่สมสมบูรณ์แบบ(Perfect Zone)

                        9. Perfect Zone (สมบูรณ์แบบ) 
                            - สถานการณ์ปัจจุบัน: นักเรียนมีความเชี่ยวชาญทั้งด้านการวิเคราะห์และด้านภาษา
                            - แผนพัฒนาขั้นต่อไป: พยายามประคับประคองให้รักษามาตรฐาน และจับตาดูอย่างใกล้ชิดถ้าหากน้องออกจากพื้นที่นี้ในอนาคต
                            - เป้าหมายต่อไป: เพิ่มเติมให้นักเรียนมีความสามารถด้านอื่นนอกจากด้านวิชาการ รวมถึงยกระดับด้านจิตใจให้อดทน ขยัน และมี winning mindset อยู่ตลอด""")
        
        with tab2:
            st.markdown("### 🏆 เปรียบเทียบอันดับ")
            fig_rank = plot_ranking_comparison(result_df, selected_alias, top_n=20)
            st.plotly_chart(fig_rank, use_container_width=True)
            
            st.info(f"💡 ลูกของคุณอยู่อันดับที่ **{target_rank}** จากทั้งหมด **{total_students}** คน")
        
        with tab3:
            st.markdown("### 📊 ข้อมูลเต็มรูปแบบ")
            
            # เลือกแสดงเฉพาะ columns ที่สำคัญ
            display_cols = ['PERSON_ID', 'RANK', 'TALENT_SCORE', 'STEM_AVG', 'LANG_AVG', 'OVERALL_AVG'] + SUBJECTS
            display_df = result_df[display_cols].copy()
            
            # Highlight นักเรียนที่เลือก
            def highlight_target(row):
                if row['PERSON_ID'] == selected_alias:
                    return ['background-color: yellow'] * len(row)
                return [''] * len(row)
            
            styled_df = display_df.style.apply(highlight_target, axis=1)
            st.dataframe(styled_df, height=400, use_container_width=True)
            
            # Download button
            csv = display_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 ดาวน์โหลดข้อมูล CSV",
                data=csv,
                file_name=f'simulation_{selected_alias}_{selected_month}.csv',
                mime='text/csv',
            )

        st.divider()
    


# =========================
if __name__ == "__main__":
    main()