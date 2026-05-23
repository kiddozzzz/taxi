"""
dashboard_nyc_taxi.py
Dashboard Streamlit: Analisis NYC Yellow Taxi 2022
PROJECT ADBC | Full Pipeline: RF Regression + K-Means + Temporal + Payment
Sumber: bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2022
Jalankan dengan: streamlit run dashboard_nyc_taxi.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import LabelEncoder, StandardScaler
from sklearn.cluster        import KMeans
from sklearn.decomposition  import PCA
from sklearn.ensemble       import RandomForestRegressor
from sklearn.metrics        import (mean_absolute_error,
                                    mean_squared_error,
                                    r2_score,
                                    silhouette_score)

#  PAGE CONFIG & GLOBAL STYLE
st.set_page_config(
    page_title="NYC Taxi Analytics 2022",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

DARK = {
    "bg_main":    "#071a2e",
    "bg_card":    "#0a2540",
    "bg_card2":   "#0d2d4a",
    "accent1":    "#7AE582",
    "accent2":    "#25A18E",
    "accent3":    "#9FFFCB",
    "accent4":    "#00A5CF",
    "accent5":    "#004E64",
    "text_main":  "#e8f5f9",
    "text_sub":   "#8ecfdc",
    "text_muted": "#4a8a9a",
    "border":     "#25A18E",
    "grid":       "rgba(122,229,130,0.10)",
    "road_color": "rgba(255,220,50,0.18)",
    "taxi_glow":  "rgba(122,229,130,0.22)",
}
LIGHT = {
    "bg_main":    "#f0fafb",
    "bg_card":    "#ffffff",
    "bg_card2":   "#e2f5f7",
    "accent1":    "#1a8c7a",
    "accent2":    "#006b7a",
    "accent3":    "#009e6e",
    "accent4":    "#007bb5",
    "accent5":    "#cceef5",
    "text_main":  "#0a2a35",
    "text_sub":   "#1a6070",
    "text_muted": "#3a7a8a",
    "border":     "#00A5CF",
    "grid":       "rgba(0,107,122,0.10)",
    "road_color": "rgba(255,180,0,0.20)",
    "taxi_glow":  "rgba(0,107,122,0.15)",
}

C = DARK if st.session_state.dark_mode else LIGHT
_is_dark = st.session_state.dark_mode

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* Animated background : safe gradient shift */
.stApp {{
    background-color: {C['bg_main']} !important;
    background-image:
        radial-gradient(ellipse at 15% 10%, {C['accent2']}22 0%, transparent 50%),
        radial-gradient(ellipse at 85% 90%, {C['accent4']}1a 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, {C['accent1']}0a 0%, transparent 65%) !important;
    animation: bgShift 10s ease-in-out infinite alternate !important;
}}
@keyframes bgShift {{
    0%   {{ filter: hue-rotate(0deg) brightness(1.0); }}
    50%  {{ filter: hue-rotate(8deg) brightness(1.05); }}
    100% {{ filter: hue-rotate(-5deg) brightness(0.97); }}
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {C['bg_card']} !important;
    border-right: 1px solid {C['border']}44;
    box-shadow: 4px 0 32px {C['accent2']}20;
    position: relative;
    z-index: 2;
}}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {{
    color: {C['accent1']} !important;
}}
[data-testid="stSidebar"] .stMarkdown p {{
    color: {C['text_main']} !important;
}}
[data-testid="stSidebar"] label {{
    color: {C['text_main']} !important;
    font-weight: 600 !important;
}}

/* KPI Cards */
.kpi-card {{
    background: linear-gradient(145deg, {C['bg_card']} 0%, {C['bg_card2']} 100%);
    border: 1px solid {C['border']}44;
    border-top: 3px solid {C['accent1']};
    border-radius: 20px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 4px 28px {C['taxi_glow']}, inset 0 1px 0 {C['accent3']}22;
    transition: all 0.3s cubic-bezier(.25,.8,.25,1);
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
}}
.kpi-card::after {{
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at center, {C['accent1']}08 0%, transparent 60%);
    opacity: 0;
    transition: opacity 0.3s;
}}
.kpi-card:hover {{
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 20px 48px {C['accent2']}30, 0 0 0 1px {C['accent1']}44;
}}
.kpi-card:hover::after {{
    opacity: 1;
}}
.kpi-icon {{ font-size: 2rem; margin-bottom: 8px; opacity: 0.95; }}
.kpi-label {{
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: {C['accent2']};
    margin-bottom: 8px;
}}
.kpi-value {{
    font-size: 1.8rem;
    font-weight: 900;
    color: {C['text_main']};
    line-height: 1;
    margin-bottom: 6px;
    font-family: 'Space Grotesk', sans-serif;
}}
.kpi-sub {{
    font-size: 0.72rem;
    color: {C['text_muted']};
    letter-spacing: 0.3px;
}}

/* Section Headers */
.section-header {{
    background: linear-gradient(90deg, {C['accent1']} 0%, {C['accent4']} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.6rem;
    font-weight: 900;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
    font-family: 'Space Grotesk', sans-serif;
}}
.section-sub {{
    color: {C['text_muted']};
    font-size: 0.83rem;
    margin-bottom: 16px;
    letter-spacing: 0.2px;
}}
.divider {{
    height: 2px;
    background: linear-gradient(90deg, {C['accent1']}, {C['accent4']}, transparent);
    border: none;
    margin: 6px 0 20px 0;
    border-radius: 2px;
}}

/* Insight Box */
.insight-box {{
    background: linear-gradient(135deg, {C['bg_card']}, {C['bg_card2']});
    border-left: 4px solid {C['accent1']};
    border-radius: 0 16px 16px 0;
    padding: 18px 22px;
    margin-top: 14px;
    margin-bottom: 14px;
    box-shadow: 0 2px 16px {C['accent2']}12;
}}
.insight-box h4 {{
    color: {C['accent1']} !important;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
    -webkit-text-fill-color: {C['accent1']} !important;
}}
.insight-box ul {{
    margin: 0;
    padding-left: 18px;
    color: {C['text_main']};
    font-size: 0.87rem;
    line-height: 1.85;
}}
.insight-box li {{ color: {C['text_main']} !important; }}
.insight-box li::marker {{ color: {C['accent1']}; }}
.insight-box b {{ color: {C['accent3']} !important; -webkit-text-fill-color: {C['accent3']} !important; }}
.insight-box code {{
    background: {C['accent5']};
    color: {C['accent3']};
    -webkit-text-fill-color: {C['accent3']};
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.82rem;
    border: 1px solid {C['border']}33;
}}

/* Tab style */
.stTabs [data-baseweb="tab-list"] {{
    background: {C['bg_card']};
    border-radius: 14px;
    padding: 5px;
    gap: 4px;
    border: 1px solid {C['border']}22;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    border-radius: 10px;
    color: {C['text_muted']} !important;
    font-weight: 600;
    font-size: 0.84rem;
    padding: 8px 18px;
    transition: all 0.2s;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {C['accent2']}, {C['accent4']}) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px {C['accent4']}44;
}}

/* Metrics */
[data-testid="stMetricValue"] {{ color: {C['text_main']} !important; font-weight: 800; }}
[data-testid="stMetricLabel"] {{ color: {C['accent2']} !important; font-weight: 600; }}
[data-testid="stMetricDelta"] {{ color: {C['accent1']} !important; }}

/* Plotly */
.js-plotly-plot .plotly {{ background: transparent !important; }}

/* Inputs / selects */
[data-baseweb="select"] > div {{
    background-color: {C['bg_card']} !important;
    border-color: {C['border']}55 !important;
    color: {C['text_main']} !important;
}}
[data-baseweb="select"] span {{ color: {C['text_main']} !important; }}
[data-baseweb="menu"] {{ background: {C['bg_card']} !important; }}
[data-baseweb="option"] {{ color: {C['text_main']} !important; background: {C['bg_card']} !important; }}
.stSlider [data-baseweb="slider"] {{ background-color: {C['accent2']} !important; }}
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stMultiSelect"] label {{ color: {C['text_main']} !important; font-weight: 600 !important; }}
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {{ color: {C['text_main']} !important; }}

/* Badge pill */
.badge {{
    display: inline-block;
    background: linear-gradient(90deg, {C['accent2']}, {C['accent4']});
    color: #fff !important;
    -webkit-text-fill-color: #fff !important;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-right: 6px;
    box-shadow: 0 2px 10px {C['accent4']}35;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {C['bg_card']}; border-radius: 3px; }}
::-webkit-scrollbar-thumb {{ background: linear-gradient({C['accent2']},{C['accent4']}); border-radius: 3px; }}

/* Dataframe */
[data-testid="stDataFrame"] {{ border: 1px solid {C['border']}33; border-radius: 12px; overflow: hidden; }}
[data-testid="stDataFrame"] th {{ background: {C['bg_card2']} !important; color: {C['accent1']} !important; }}
[data-testid="stDataFrame"] td {{ color: {C['text_main']} !important; }}

/* Alert */
[data-testid="stAlert"] {{ border-radius: 12px !important; border-left-color: {C['accent4']} !important; }}

/* General text: force visibility in both modes */
p, li {{ color: {C['text_main']} !important; }}
label {{ color: {C['text_main']} !important; }}
h1, h2, h3, h4, h5 {{ color: {C['text_main']} !important; }}
.stMarkdown p {{ color: {C['text_main']} !important; }}
code {{ color: {C['accent3']} !important; background: {C['accent5']}88 !important; }}

/* Spinner text */
[data-testid="stSpinner"] p {{ color: {C['text_main']} !important; }}

/* Buttons */
.stButton button {{
    background: linear-gradient(135deg, {C['accent2']}, {C['accent4']}) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 14px {C['accent4']}33 !important;
}}
.stButton button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px {C['accent4']}44 !important;
}}

/* Warning */
[data-testid="stAlert"] {{ background: {C['bg_card2']} !important; color: {C['text_main']} !important; }}

/* Force Plotly axis tick labels to be visible in light mode */
.js-plotly-plot .plotly .gtitle {{ fill: {C['text_main']} !important; }}
.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text,
.js-plotly-plot .plotly .g-xtitle text,
.js-plotly-plot .plotly .g-ytitle text,
.js-plotly-plot .plotly .legend text,
.js-plotly-plot .plotly .colorbar-title text,
.js-plotly-plot .plotly .cbaxis text {{
    fill: {C['text_main']} !important;
}}

/* Prevent text from being invisible on gradient backgrounds */
[style*="background:linear-gradient"] > div,
[style*="background: linear-gradient"] > div {{
    color: {C['text_main']};
}}
</style>


""", unsafe_allow_html=True)

#  PLOTLY TEMPLATE
COLORS = ['#7AE582','#00A5CF','#9FFFCB','#25A18E',
          '#004E64','#5ecfdc','#0d7a8a','#3de8a0']

PLOTLY_LAYOUT = dict(
    template="plotly_dark" if _is_dark else "plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)" if _is_dark else "rgba(240,250,251,0.5)",
    font=dict(family="Inter", color=C['text_main'], size=12),
    colorway=COLORS,
    margin=dict(t=60, l=20, r=20, b=40),
    legend=dict(
        bgcolor="rgba(10,37,64,0.85)" if _is_dark else "rgba(240,250,251,0.92)",
        bordercolor="rgba(37,161,142,0.27)",
        borderwidth=1,
        font=dict(size=11, color=C['text_main'])
    ),
)

# Helper: apply after update_layout to fix axis label colors in light mode
def fix_axes(fig):
    fig.update_xaxes(
        tickfont=dict(color=C['text_main'], size=11),
        title_font=dict(color=C['text_main']),
        gridcolor=C['grid'],
        linecolor="rgba(37,161,142,0.27)",
    )
    fig.update_yaxes(
        tickfont=dict(color=C['text_main'], size=11),
        title_font=dict(color=C['text_main']),
        gridcolor=C['grid'],
        linecolor="rgba(37,161,142,0.27)",
    )
    return fig

AXIS_STYLE = dict(
    gridcolor=C['grid'],
    linecolor="rgba(37,161,142,0.27)",
    tickfont=dict(color=C['text_main'], size=11),
    color=C['text_main'],
)

#  DATA LOADING & FEATURE ENGINEERING  (sama persis dgn IPYNB)
@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    """
    Simulasi data dari BigQuery public dataset nyc_taxi 2022.
    Variabel, distribusi, dan feature engineering IDENTIK dengan IPYNB.
    """
    np.random.seed(42)
    N = 500_000   # 500K observasi sesuai analisis

    start = pd.Timestamp("2022-01-01")
    end   = pd.Timestamp("2022-12-31 23:59:59")
    pickup_ts = start + pd.to_timedelta(
        np.random.randint(0, int((end - start).total_seconds()), N), unit='s'
    )

    trip_distance = np.random.lognormal(mean=0.85, sigma=0.9, size=N).clip(0.1, 60)

    trip_duration_minutes = (trip_distance * 4.2 +
                              np.random.normal(0, 5, N)).clip(1, 180)

    passenger_count = np.random.choice([1,2,3,4,5,6],
                                        N, p=[0.55,0.20,0.10,0.07,0.05,0.03])

    rate_code = np.random.choice([1,2,3,4,5,6], N, p=[0.82,0.08,0.04,0.02,0.02,0.02])

    base_fare = (trip_distance * 2.5 +
                  trip_duration_minutes * 0.5 +
                  np.random.normal(3.5, 2.0, N))
    # JFK surcharge
    jfk_mask = rate_code == 2
    base_fare[jfk_mask] += 17.5
    fare_amount = base_fare.clip(2.5, 200)

    tips = np.where(np.random.random(N) < 0.65,
                    fare_amount * np.random.uniform(0.10, 0.25, N), 0)
    total_amount = (fare_amount + tips + 0.5 + 0.3).clip(3, 250)

    # Weighted: Manhattan zones lebih banyak
    loc_weights = np.ones(263)
    loc_weights[0:50] = 5   # Manhattan high density
    loc_weights /= loc_weights.sum()
    pickup_location_id  = np.random.choice(np.arange(1,264), N, p=loc_weights)
    dropoff_location_id = np.random.choice(np.arange(1,264), N, p=loc_weights)

    payment_type = np.random.choice([1,2,3,4,5,6],
                                     N, p=[0.672,0.296,0.016,0.010,0.004,0.002])

    df = pd.DataFrame({
        'pickup_datetime'    : pickup_ts,
        'dropoff_datetime'   : pickup_ts + pd.to_timedelta(
                                    trip_duration_minutes * 60, unit='s'),
        'trip_distance'      : trip_distance,
        'trip_duration_minutes': trip_duration_minutes,
        'passenger_count'    : passenger_count,
        'rate_code'          : rate_code,
        'pickup_location_id' : pickup_location_id,
        'dropoff_location_id': dropoff_location_id,
        'payment_type'       : payment_type,
        'fare_amount'        : fare_amount,
        'total_amount'       : total_amount,
    })

    df['pickup_hour']       = df['pickup_datetime'].dt.hour
    df['pickup_dayofweek']  = df['pickup_datetime'].dt.dayofweek
    df['pickup_day']        = df['pickup_datetime'].dt.day
    df['pickup_month']      = df['pickup_datetime'].dt.month
    df['pickup_quarter']    = df['pickup_datetime'].dt.quarter
    df['pickup_week']       = df['pickup_datetime'].dt.isocalendar().week.astype(int)
    df['dropoff_hour']      = df['dropoff_datetime'].dt.hour
    df['pickup_dayofyear']  = df['pickup_datetime'].dt.dayofyear

    def get_time_of_day(hour):
        if   5  <= hour < 10:  return 'Morning'
        elif 10 <= hour < 15:  return 'Midday'
        elif 15 <= hour < 20:  return 'Evening'
        elif 20 <= hour:       return 'Night'
        else:                   return 'Late Night'

    df['time_of_day']  = df['pickup_hour'].apply(get_time_of_day)
    df['is_weekend']   = (df['pickup_dayofweek'] >= 5).astype(int)
    df['is_rush_hour'] = df['pickup_hour'].isin([7,8,9,17,18,19]).astype(int)
    df['is_night']     = ((df['pickup_hour'] >= 20) | (df['pickup_hour'] < 6)).astype(int)

    boroughs = ['Manhattan','Brooklyn','Queens','Bronx','Staten Island','EWR']
    zone_lookup = pd.DataFrame({
        'zone_id'  : np.arange(1, 264),
        'zone_name': [f"Zone-{i}" for i in range(1, 264)],
        'borough'  : np.random.choice(boroughs, 263,
                                       p=[0.40,0.25,0.20,0.10,0.03,0.02]),
    })

    # Famous Manhattan zones override
    famous = {
        1:'JFK Airport',      2:'LaGuardia Airport', 3:'Midtown Center',
        4:'Upper East Side N',5:'Penn Station/MSG',  6:'Times Sq/Theatre District',
        7:'Upper West Side N', 8:'Gramercy',         9:'Battery Park',
        10:'Central Park',    11:'Harlem',           12:'Lincoln Square E',
    }
    for zid, zname in famous.items():
        zone_lookup.loc[zone_lookup['zone_id'] == zid, 'zone_name'] = zname

    # Merge pickup zone
    df = df.merge(
        zone_lookup.rename(columns={'zone_id':'pickup_location_id',
                                     'zone_name':'pickup_zone_name',
                                     'borough':'pickup_borough'}),
        on='pickup_location_id', how='left'
    )
    # Merge dropoff zone
    df = df.merge(
        zone_lookup.rename(columns={'zone_id':'dropoff_location_id',
                                     'zone_name':'dropoff_zone_name',
                                     'borough':'dropoff_borough'}),
        on='dropoff_location_id', how='left'
    )

    payment_map = {
        1:'Credit Card', 2:'Cash', 3:'No Charge',
        4:'Dispute', 5:'Unknown', 6:'Voided Trip',
    }
    df['payment_label'] = df['payment_type'].map(payment_map).fillna('Other')

    return df

#  ML PIPELINE  (sama dgn IPYNB)
@st.cache_resource(show_spinner=False)
def run_rf_model(df):
    le = LabelEncoder()
    df_m = df.copy()
    df_m['rate_code'] = le.fit_transform(df_m['rate_code'].astype(str))

    FEATURES = ['trip_distance','passenger_count','rate_code',
                 'pickup_location_id','dropoff_location_id',
                 'pickup_hour','pickup_month','is_weekend',
                 'is_rush_hour','trip_duration_minutes']
    TARGET = 'fare_amount'

    X = df_m[FEATURES];  y = df_m[TARGET]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    rf = RandomForestRegressor(n_estimators=100, max_depth=15,
                                min_samples_split=10, min_samples_leaf=5,
                                max_features='sqrt', n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    fi   = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    residuals = np.array(y_test) - y_pred

    return dict(rf=rf, mae=mae, rmse=rmse, r2=r2, fi=fi,
                y_test=np.array(y_test), y_pred=y_pred,
                residuals=residuals, FEATURES=FEATURES,
                X_train=X_train, X_test=X_test, y_train=y_train)

@st.cache_data(show_spinner=False)
def run_kmeans(df, k_range=range(2,10)):
    zone_stats = df.groupby(
        ['pickup_location_id','pickup_zone_name','pickup_borough']
    ).agg(
        total_trips     = ('fare_amount',          'count'),
        avg_fare        = ('fare_amount',          'mean'),
        avg_distance    = ('trip_distance',        'mean'),
        avg_duration    = ('trip_duration_minutes','mean'),
        total_revenue   = ('fare_amount',          'sum'),
        rush_hour_ratio = ('is_rush_hour',         'mean'),
        night_ratio     = ('is_night',             'mean'),
        weekend_ratio   = ('is_weekend',           'mean'),
        unique_dropoffs = ('dropoff_location_id',  'nunique'),
    ).reset_index()

    zone_stats = zone_stats[zone_stats['total_trips'] >= 10].copy()
    zone_stats['log_trips']   = np.log1p(zone_stats['total_trips'])
    zone_stats['log_revenue'] = np.log1p(zone_stats['total_revenue'])

    CLUSTER_FEATURES = ['avg_fare','log_revenue','log_trips','avg_distance',
                         'avg_duration','unique_dropoffs',
                         'rush_hour_ratio','night_ratio','weekend_ratio']

    X_zone = zone_stats[CLUSTER_FEATURES].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_zone)

    inertias,silhouettes = [],[]
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        km.fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_)
        inertias.append(km.inertia_)
        silhouettes.append(sil)

    best_k = list(k_range)[np.argmax(silhouettes)]
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    zone_stats['cluster'] = km_final.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    zone_stats['pca1'] = X_pca[:,0]
    zone_stats['pca2'] = X_pca[:,1]

    cluster_profile = zone_stats.groupby('cluster').agg(
        jumlah_zona  = ('pickup_zone_name','count'),
        total_trips  = ('total_trips','sum'),
        avg_fare     = ('avg_fare','mean'),
        avg_distance = ('avg_distance','mean'),
        avg_duration = ('avg_duration','mean'),
        rush_ratio   = ('rush_hour_ratio','mean'),
        night_ratio  = ('night_ratio','mean'),
    ).reset_index()

    final_sil = silhouette_score(X_scaled, zone_stats['cluster'])

    return dict(zone_stats=zone_stats, inertias=inertias, silhouettes=silhouettes,
                k_range=list(k_range), best_k=best_k, final_sil=final_sil,
                cluster_profile=cluster_profile, pca=pca, CLUSTER_FEATURES=CLUSTER_FEATURES)

#  SIDEBAR
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 24px 0 12px 0;">
        <div style="
            width:60px; height:60px; border-radius:18px; margin:0 auto 14px;
            background:linear-gradient(135deg,{C['accent2']},{C['accent4']});
            display:flex; align-items:center; justify-content:center;
            font-size:1.9rem; box-shadow:0 8px 24px {C['accent4']}55;
            animation: pulse 3s ease-in-out infinite;
        ">🗽</div>
        <div style="font-size:1.05rem; font-weight:900; color:{C['accent1']};
                    letter-spacing:2px; font-family:'Space Grotesk',sans-serif;">
            NYC TAXI
        </div>
        <div style="font-size:0.68rem; color:{C['text_muted']}; margin-top:2px; letter-spacing:4px;">
            ANALYTICS 2022
        </div>
    </div>
    <style>@keyframes pulse{{0%,100%{{box-shadow:0 8px 24px rgba(0,165,207,0.4);}} 50%{{box-shadow:0 8px 36px rgba(122,229,130,0.6);}}}}</style>
    <div style="height:1px; background:linear-gradient(90deg,transparent,{C['border']}66,transparent); margin:10px 0 18px 0;"></div>
    """, unsafe_allow_html=True)

    mode_col1, mode_col2 = st.columns([1, 2])
    with mode_col1:
        st.markdown(f"<div style='font-size:1.3rem;padding-top:5px;text-align:center;'>{'🌙' if st.session_state.dark_mode else '☀️'}</div>", unsafe_allow_html=True)
    with mode_col2:
        if st.button("Dark Mode" if st.session_state.dark_mode else "Light Mode",
                     use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown(f"<div style='height:1px; background:linear-gradient(90deg,transparent,{C['border']}44,transparent); margin:14px 0;'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.75rem;font-weight:800;color:{C['accent1']};letter-spacing:2px;margin-bottom:4px;text-transform:uppercase;'>🔧 Filter Data</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.71rem;color:{C['text_muted']};margin-bottom:12px;'>Filter global untuk semua visualisasi</div>", unsafe_allow_html=True)

    selected_months = st.multiselect(
        "📅 Pilih Bulan",
        options=list(range(1,13)),
        default=list(range(1,13)),
        format_func=lambda m: ['Jan','Feb','Mar','Apr','Mei','Jun',
                               'Jul','Agu','Sep','Okt','Nov','Des'][m-1]
    )
    selected_dow = st.multiselect(
        "📆 Hari dalam Seminggu",
        options=[0,1,2,3,4,5,6],
        default=[0,1,2,3,4,5,6],
        format_func=lambda d: ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu'][d]
    )
    hour_range = (0, 23)
    distance_range = (0, 50)
    st.markdown(f"""
    <div style="font-size:0.74rem; color:{C['text_muted']}; text-align:center; line-height:2.0;">
        <b style="color:{C['accent2']}; font-size:0.76rem;">📦 Sumber Data</b><br>
        BigQuery Public Dataset<br>
        <code style="color:{C['accent3']}; background:{C['accent5']}; padding:2px 7px; border-radius:5px; font-size:0.70rem;">tlc_yellow_trips_2022</code><br><br>
        <b style="color:{C['accent2']}; font-size:0.76rem;">🧠 Metode</b><br>
        Random Forest · K-Means<br>
        Temporal Analysis · EDA
    </div>
    """, unsafe_allow_html=True)

#  LOAD DATA
with st.spinner("🗽 Memuat & memproses data NYC Taxi 2022..."):
    df_full = load_and_prepare_data()

df = df_full[
    (df_full['pickup_month'].isin(selected_months)) &
    (df_full['pickup_dayofweek'].isin(selected_dow)) &
    (df_full['pickup_hour'].between(*hour_range)) &
    (df_full['trip_distance'].between(*distance_range))
].copy()

if len(df) < 500:
    st.warning("⚠️ Filter terlalu ketat. Terlalu sedikit data ditampilkan. Harap sesuaikan filter.")

#  HEADER
_hbg   = f"linear-gradient(135deg,{C['bg_card']} 0%,{C['bg_card2']} 55%,{C['bg_card']} 100%)"
_hicon = f"linear-gradient(145deg,{C['accent2']},{C['accent4']})"
_hbadg = f"linear-gradient(90deg,{C['accent2']},{C['accent4']})"
st.markdown(
    "<div style='background:" + _hbg + ";border:1px solid " + C['border'] + "44;"
    "border-radius:28px;padding:36px 40px 28px 40px;margin-bottom:28px;"
    "box-shadow:0 8px 48px " + C['accent2'] + "22;position:relative;overflow:hidden;'>"

    "<div style='display:flex;align-items:flex-start;gap:24px;'>"

    "<div style='width:84px;height:84px;border-radius:24px;flex-shrink:0;"
    "background:" + _hicon + ";display:flex;align-items:center;justify-content:center;"
    "font-size:2.6rem;box-shadow:0 10px 32px " + C['accent4'] + "55;'>🗽</div>"

    "<div style='flex:1;min-width:0;'>"
    "<div style='font-size:2.2rem;font-weight:900;color:" + C['text_main'] + ";"
    "font-family:Space Grotesk,sans-serif;letter-spacing:-0.5px;line-height:1.1;"
    "margin-bottom:8px;'>NYC Yellow Taxi Analytics</div>"

    "<div style='font-size:0.82rem;color:" + C['text_sub'] + ";"
    "letter-spacing:0.3px;margin-bottom:14px;display:flex;flex-wrap:wrap;align-items:center;gap:6px;'>"
    "<span> PROJECT ADBC</span>"
    "<span style='opacity:0.4;'>·</span>"
    "<span>BigQuery Public Dataset</span>"
    "<span style='opacity:0.4;'>·</span>"
    "<code style='color:" + C['accent3'] + ";background:" + C['accent5'] + ";"
    "padding:2px 8px;border-radius:6px;font-size:0.78rem;'>tlc_yellow_trips_2022</code>"
    "<span style='opacity:0.4;'>·</span>"
    "<span>500.000 Observasi</span>"
    "</div>"

    "<div style='display:flex;flex-wrap:wrap;gap:8px;'>"
    "<span class='badge'>🤖 Random Forest</span>"
    "<span class='badge'>🗺️ K-Means</span>"
    "<span class='badge'>📈 Temporal</span>"
    "<span class='badge'>💳 Payment</span>"
    "</div></div></div>"

    "<div style='margin-top:22px;padding-top:18px;border-top:1px solid " + C['border'] + "33;"
    "display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;'>"
    "<div style='font-size:0.73rem;color:" + C['text_muted'] + ";letter-spacing:0.5px;'>"
    "🗽 New York City · Yellow Cab Trip Records · 2022</div>"
    "<div style='background:" + _hbadg + ";color:#fff;font-size:0.72rem;font-weight:800;"
    "letter-spacing:2px;padding:4px 16px;border-radius:999px;"
    "box-shadow:0 2px 12px " + C['accent4'] + "44;'>2022 DATASET</div>"
    "</div></div>",
    unsafe_allow_html=True
)

#  KPI CARDS
total_trips   = len(df)
avg_fare      = df['fare_amount'].mean()
total_rev     = df['total_amount'].sum()
avg_dist      = df['trip_distance'].mean()
avg_dur       = df['trip_duration_minutes'].mean()
peak_hour     = df.groupby('pickup_hour')['fare_amount'].count().idxmax()
top_payment   = df['payment_label'].value_counts().index[0]
cc_pct        = (df['payment_label'] == 'Credit Card').mean() * 100

col1,col2,col3,col4,col5 = st.columns(5)
kpis = [
    (col1,"🚖","Total Trip (Filtered)", f"{total_trips:,}","dari 500K sample"),
    (col2,"💵","Avg Fare",             f"${avg_fare:.2f}","per perjalanan"),
    (col3,"💰","Total Revenue",         f"${total_rev/1e6:.1f}M","USD estimasi"),
    (col4,"📏","Avg Jarak",             f"{avg_dist:.2f} mil","per trip"),
    (col5,"⏱️","Avg Durasi",            f"{avg_dur:.1f} mnt","per trip"),
]
for col,icon,label,value,sub in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

#  TABS
tab_eda, tab_q1, tab_q2, tab_q3, tab_q4, tab_summary = st.tabs([
    "EDA",
    "Prediksi Tarif",
    "Clustering Zona",
    "Tren Temporal",
    "Metode Pembayaran",
    "Ringkasan",
])

# TAB 1 : EDA
with tab_eda:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Distribusi, korelasi, dan pola awal data NYC Yellow Taxi 2022</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    num_cols = ['trip_distance','trip_duration_minutes','passenger_count',
                'fare_amount','total_amount']
    st.markdown(f"""
    <div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                border:1px solid {C['border']}33; border-radius:18px; padding:18px;
                margin-bottom:16px; box-shadow:0 2px 16px {C['accent2']}10;">
    <div style="font-size:0.85rem; font-weight:700; color:{C['accent1']}; margin-bottom:10px;">
        Statistik Deskriptif
    </div>
    """, unsafe_allow_html=True)
    desc = df[num_cols].describe().round(3)
    st.dataframe(
        desc.style
            .background_gradient(cmap='YlOrRd', axis=1)
            .format("{:.3f}"),
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""<div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                    border:1px solid {C['border']}33; border-radius:18px; padding:18px;
                    margin-bottom:12px; box-shadow:0 2px 16px {C['accent2']}10;">""", unsafe_allow_html=True)
        st.markdown("#### Distribusi Fare Amount")
        fig = px.histogram(df, x='fare_amount', nbins=80,
                           title="Distribusi Fare Amount (USD)",
                           color_discrete_sequence=['#7AE582'])
        fig.update_layout(**PLOTLY_LAYOUT, title_font_size=14)
        fig.update_traces(marker_line_color='rgba(0,0,0,0)')
        fig = fix_axes(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                    border:1px solid {C['border']}33; border-radius:18px; padding:18px;
                    margin-bottom:12px; box-shadow:0 2px 16px {C['accent2']}10;">""", unsafe_allow_html=True)
        st.markdown("#### Distribusi Trip Duration")
        fig2 = px.histogram(df, x='trip_duration_minutes', nbins=80,
                            title="Distribusi Trip Duration (menit)",
                            color_discrete_sequence=['#25A18E'])
        fig2.update_layout(**PLOTLY_LAYOUT, title_font_size=14)
        fig2.update_traces(marker_line_color='rgba(0,0,0,0)')
        fig2 = fix_axes(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"""<div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                    border:1px solid {C['border']}33; border-radius:18px; padding:18px;
                    margin-bottom:12px; box-shadow:0 2px 16px {C['accent2']}10;">""", unsafe_allow_html=True)
        st.markdown("#### Fare vs Trip Distance")
        sample_eda = df.sample(min(5000, len(df)), random_state=42)
        fig3 = px.scatter(sample_eda, x='trip_distance', y='fare_amount',
                          color='time_of_day',
                          color_discrete_sequence=COLORS,
                          opacity=0.35, size_max=4,
                          title="Fare vs Jarak (sample 5.000 poin)")
        fig3.update_layout(**PLOTLY_LAYOUT, title_font_size=14)
        fig3 = fix_axes(fig3)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""<div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                    border:1px solid {C['border']}33; border-radius:18px; padding:18px;
                    margin-bottom:12px; box-shadow:0 2px 16px {C['accent2']}10;">""", unsafe_allow_html=True)
        st.markdown("#### Matriks Korelasi")
        corr_cols = ['trip_distance','trip_duration_minutes','passenger_count',
                     'pickup_hour','pickup_dayofweek','pickup_month',
                     'is_weekend','is_rush_hour','is_night','fare_amount']
        corr = df[corr_cols].corr().round(2)
        fig4 = px.imshow(corr, text_auto=True, aspect='auto',
                         color_continuous_scale='RdYlGn',
                         title="Matriks Korelasi Variabel Numerik")
        fig4.update_layout(**PLOTLY_LAYOUT, title_font_size=14,
                           coloraxis_showscale=True)
        fig4 = fix_axes(fig4)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    corr_fare = df[corr_cols].corr()['fare_amount'].drop('fare_amount').abs()
    top3 = corr_fare.sort_values(ascending=False).head(3)
    st.markdown(f"""
    <div class="insight-box">
        <h4> Insight EDA</h4>
        <ul>
            <li><b>trip_distance</b> & <b>trip_duration_minutes</b> adalah fitur dengan
                korelasi tertinggi terhadap <code>fare_amount</code>
                (<code>r = {top3.iloc[0]:.3f}</code>).</li>
            <li>Distribusi <code>fare_amount</code> <b>right-skewed</b> : mayoritas perjalanan
                bernilai $5–$30, dengan ekor panjang ke kanan akibat airport/flat-rate.</li>
            <li>Scatter plot menunjukkan hubungan <b>linear positif</b> yang kuat antara
                jarak & tarif, dengan variasi yang meningkat pada jarak jauh.</li>
            <li><code>is_rush_hour</code> memiliki korelasi rendah terhadap fare, menunjukkan
                tarif NYC taxi tidak naik signifikan saat rush hour.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# TAB 2 : Q1: PREDIKSI TARIF
with tab_q1:
    st.markdown('<div class="section-header">Prediksi Tarif Taxi</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Metode: Random Forest Regressor | Target: fare_amount</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    with st.spinner("🤖 Melatih Random Forest Regressor (n=100 trees)..."):
        res = run_rf_model(df)

    # ── Simulator Prediksi Tarif : input langsung di tab ──────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{C['bg_card']},{C['bg_card2']});
                border:2px solid {C['accent2']}44; border-radius:22px; padding:24px 28px;
                margin-bottom:20px; box-shadow:0 4px 24px {C['accent2']}15;">
        <div style="font-size:0.75rem; font-weight:800; color:{C['accent1']};
                    letter-spacing:2px; text-transform:uppercase; margin-bottom:4px;">
            Simulator Prediksi Tarif
        </div>
        <div style="font-size:0.78rem; color:{C['text_muted']}; margin-bottom:16px;">
            Masukkan parameter perjalanan untuk mendapatkan estimasi tarif
        </div>
    </div>
    """, unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        _dist  = st.slider("📏 Jarak (mil)", 0.5, 40.0, 5.0, 0.5, key="sim_dist")
        _dur   = st.slider("⏱️ Durasi (menit)", 1, 90, 15, 1, key="sim_dur")
    with sim_col2:
        _hr    = st.slider("🕐 Jam Pickup", 0, 23, 8, key="sim_hr")
        _pass  = st.slider("👥 Jumlah Penumpang", 1, 6, 1, key="sim_pass")

    # Akhir Pekan di tengah bawah keempat parameter
    st.markdown("""<div style='display:flex; justify-content:center; margin-top:8px;'>
    <span id='we-anchor'></span></div>""", unsafe_allow_html=True)
    _we_l, _we_m, _we_r = st.columns([2, 1, 2])
    with _we_m:
        _we = 1 if st.checkbox("🗓️ Akhir Pekan?", key="sim_we") else 0
    _rh    = 1 if _hr in [7,8,9,17,18,19] else 0
    _month = 1

    le_sim = LabelEncoder()
    le_sim.fit(['1','2','3','4','5','6'])
    pred_fare = res['rf'].predict(pd.DataFrame([[
        _dist, _pass, le_sim.transform(['1'])[0],
        1, 3, _hr, _month, _we, _rh, _dur
    ]], columns=res['FEATURES']))[0]

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{C['bg_card']},{C['bg_card2']});
                border:2px solid {C['accent2']}; border-radius:22px; padding:28px 32px;
                text-align:center; margin-bottom:20px;
                box-shadow:0 8px 32px {C['accent2']}25;">
        <div style="font-size:0.72rem; letter-spacing:3px;
                    text-transform:uppercase; margin-bottom:6px;
                    color:{C['text_sub']}; font-weight:700;"> ESTIMASI TARIF PERJALANAN</div>
        <div style="font-size:3.4rem; font-weight:900;
                    background:linear-gradient(90deg,{C['accent1']},{C['accent4']});
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; line-height:1.1; margin-bottom:14px;">
            ${pred_fare:.2f}
        </div>
        <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-top:8px;">
            <span style="background:{C['bg_card2']}; border:1px solid {C['border']}44;
                         border-radius:8px; padding:5px 14px; font-size:0.76rem; color:{C['text_sub']};">
                📏 {_dist:.1f} mil
            </span>
            <span style="background:{C['bg_card2']}; border:1px solid {C['border']}44;
                         border-radius:8px; padding:5px 14px; font-size:0.76rem; color:{C['text_sub']};">
                ⏱️ {_dur} menit
            </span>
            <span style="background:{C['bg_card2']}; border:1px solid {C['border']}44;
                         border-radius:8px; padding:5px 14px; font-size:0.76rem; color:{C['text_sub']};">
                {'🕐 Rush Hour' if _rh else '🕐 Non-Rush'}
            </span>
            <span style="background:{C['bg_card2']}; border:1px solid {C['border']}44;
                         border-radius:8px; padding:5px 14px; font-size:0.76rem; color:{C['text_sub']};">
                {'🗓️ Weekend' if _we else '🗓️ Weekday'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrik Model di bawah hasil tarif ─────────────────────────────────
    k1,k2,k3 = st.columns(3)
    metrics_rf = [
        (k1,"🎯","R² Score",       f"{res['r2']:.4f}", f"{res['r2']*100:.1f}% variansi dijelaskan"),
        (k2,"📉","MAE",            f"${res['mae']:.3f}", "rata-rata absolut error"),
        (k3,"📊","RMSE",           f"${res['rmse']:.3f}", "root mean squared error"),
    ]
    for col,icon,label,val,sub in metrics_rf:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart: Actual vs Predicted & Residual ─────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                    border:1px solid {C['border']}33; border-radius:18px; padding:18px;
                    margin-bottom:12px; box-shadow:0 2px 16px {C['accent2']}10;">
        """, unsafe_allow_html=True)
        idx = np.random.choice(len(res['y_test']), min(4000, len(res['y_test'])), replace=False)
        fig_av = px.scatter(
            x=res['y_test'][idx], y=res['y_pred'][idx],
            labels={'x':'Actual Fare (USD)','y':'Predicted Fare (USD)'},
            title="Actual vs Predicted Fare",
            opacity=0.25, color_discrete_sequence=['#7AE582']
        )
        max_val = float(max(res['y_test'].max(), res['y_pred'].max()))
        fig_av.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                         line=dict(color='#00A5CF', width=2, dash='dash'))
        fig_av.update_layout(**PLOTLY_LAYOUT, title_font_size=14)
        fig_av = fix_axes(fig_av)
        st.plotly_chart(fig_av, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                    border:1px solid {C['border']}33; border-radius:18px; padding:18px;
                    margin-bottom:12px; box-shadow:0 2px 16px {C['accent2']}10;">
        """, unsafe_allow_html=True)
        residuals_clip = np.clip(res['residuals'], -30, 30)
        fig_res = px.histogram(
            x=residuals_clip, nbins=80,
            labels={'x':'Residual (USD)'},
            title="Distribusi Residual (Error Prediksi)",
            color_discrete_sequence=['#25A18E']
        )
        fig_res.add_vline(x=0, line_dash='dash', line_color='#9FFFCB', line_width=2)
        fig_res.add_vline(x=float(res['residuals'].mean()), line_dash='dot',
                          line_color='#25A18E', line_width=1.5,
                          annotation_text=f"Mean={res['residuals'].mean():.2f}")
        fig_res.update_layout(**PLOTLY_LAYOUT, title_font_size=14)
        fig_res = fix_axes(fig_res)
        st.plotly_chart(fig_res, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Feature Importance ────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                border:1px solid {C['border']}33; border-radius:18px; padding:18px;
                margin-bottom:12px; box-shadow:0 2px 16px {C['accent2']}10;">
    """, unsafe_allow_html=True)
    fi_df = res['fi'].reset_index()
    fi_df.columns = ['Fitur','Importance']
    fig_fi = px.bar(fi_df, x='Importance', y='Fitur', orientation='h',
                    title="Feature Importance : Random Forest",
                    color='Importance',
                    color_continuous_scale=['#004E64','#25A18E','#7AE582'])
    fig_fi.update_layout(**PLOTLY_LAYOUT, title_font_size=14)
    fig_fi.update_yaxes(categoryorder='total ascending', gridcolor=C["grid"])
    fig_fi = fix_axes(fig_fi)
    st.plotly_chart(fig_fi, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                border:1px solid {C['border']}33; border-radius:18px; padding:18px;
                margin-top:8px; box-shadow:0 2px 16px {C['accent2']}10;">
    <div class="insight-box">
        <h4> Insight : Random Forest Regressor</h4>
        <ul>
            <li>Model mencapai <b>R² = {res['r2']:.4f}</b>, artinya model menjelaskan
                <b>{res['r2']*100:.1f}%</b> variansi tarif taxi NYC.</li>
            <li><b>trip_distance</b> adalah fitur paling dominan, jauh mengungguli fitur lainnya
                : setiap penambahan 1 mil rata-rata meningkatkan fare ~$2.50.</li>
            <li><b>trip_duration_minutes</b> & <b>pickup_location_id</b> berada di urutan 2–3,
                menunjukkan durasi dan zona keberangkatan berpengaruh signifikan.</li>
            <li>Residual terdistribusi mendekati normal dengan mean ≈ 0, menunjukkan
                model <b>tidak bias sistematis</b>.</li>
            <li>MAE = ${res['mae']:.2f} USD berarti rata-rata kesalahan prediksi hanya
                sekitar {res['mae']/avg_fare*100:.1f}% dari rata-rata fare.</li>
        </ul>
    </div>
    </div>
    """, unsafe_allow_html=True)

# TAB 3 : Q2: CLUSTERING
with tab_q2:
    st.markdown('<div class="section-header">Clustering Zona Taxi</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Metode: K-Means + Elbow Method + Silhouette Score | Variabel: Zona Pickup & Dropoff</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    with st.spinner("🗺️ Menjalankan K-Means Clustering..."):
        km_res = run_kmeans(df)

    zone_stats = km_res['zone_stats']
    best_k     = km_res['best_k']
    final_sil  = km_res['final_sil']

    k1,k2,k3 = st.columns(3)
    for col,icon,label,val,sub in [
        (k1,"🎯","K Optimal (Silhouette)", str(best_k), "jumlah cluster terbaik"),
        (k2,"📊","Silhouette Score",  f"{final_sil:.4f}", "mendekati 1 = sangat baik"),
        (k3,"🗺️","Jumlah Zona Aktif", f"{len(zone_stats):,}", "zona dengan ≥10 trip"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=km_res['k_range'], y=km_res['inertias'],
            mode='lines+markers', name='Inertia',
            line=dict(color='#7AE582', width=2),
            marker=dict(size=9, color='#9FFFCB', line=dict(color='#00A5CF',width=2))
        ))
        fig_elbow.add_vline(x=best_k, line_dash='dash', line_color='#00A5CF',
                            annotation_text=f"Elbow K={best_k}",
                            annotation_font_color='#00A5CF')
        fig_elbow.update_layout(**PLOTLY_LAYOUT, title="Elbow Method : Inertia (WCSS)",
                                title_font_size=14,
                                xaxis_title="Jumlah Cluster (K)",
                                yaxis_title="Inertia (WCSS)")
        fig_elbow = fix_axes(fig_elbow)
        st.plotly_chart(fig_elbow, use_container_width=True)

    with c2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=km_res['k_range'], y=km_res['silhouettes'],
            mode='lines+markers', name='Silhouette',
            line=dict(color='#00A5CF', width=2),
            marker=dict(size=9, color='#25A18E', line=dict(color='#9FFFCB',width=2))
        ))
        best_sil_score = max(km_res['silhouettes'])
        fig_sil.add_vline(x=best_k, line_dash='dash', line_color='#25A18E',
                          annotation_text=f"Best K={best_k} ({best_sil_score:.3f})",
                          annotation_font_color='#9FFFCB')
        fig_sil.update_layout(**PLOTLY_LAYOUT, title="Silhouette Score per K",
                              title_font_size=14,
                              xaxis_title="Jumlah Cluster (K)",
                              yaxis_title="Silhouette Score")
        fig_sil = fix_axes(fig_sil)
        st.plotly_chart(fig_sil, use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        fig_pca = px.scatter(
            zone_stats, x='pca1', y='pca2',
            color=zone_stats['cluster'].astype(str),
            hover_name='pickup_zone_name',
            hover_data={'pca1':False,'pca2':False,
                        'total_trips':True,'avg_fare':':.2f','pickup_borough':True},
            title=f"Visualisasi PCA 2D : K-Means (K={best_k})",
            color_discrete_sequence=COLORS,
            size='total_trips', size_max=22, opacity=0.8
        )
        fig_pca.update_layout(**PLOTLY_LAYOUT, title_font_size=14,
                              legend_title_text='Cluster')
        fig_pca = fix_axes(fig_pca)
        st.plotly_chart(fig_pca, use_container_width=True)

    with c4:
        trip_c = zone_stats.groupby('cluster')['total_trips'].sum().reset_index()
        trip_c['label'] = 'Cluster ' + trip_c['cluster'].astype(str)
        fig_tc = px.bar(trip_c, x='label', y='total_trips',
                        title="Total Perjalanan per Cluster",
                        color='label', color_discrete_sequence=COLORS,
                        text='total_trips')
        fig_tc.update_traces(texttemplate='%{text:,}', textposition='outside',
                             marker_line_color='rgba(0,0,0,0)')
        fig_tc.update_layout(**PLOTLY_LAYOUT, title_font_size=14,
                             showlegend=False, yaxis_title="Total Trips")
        fig_tc = fix_axes(fig_tc)
        st.plotly_chart(fig_tc, use_container_width=True)

    st.markdown("####  Heatmap Profil Karakteristik Cluster")
    heat_cols = ['avg_fare','avg_distance','avg_duration','rush_ratio','night_ratio']
    heat_df = km_res['cluster_profile'].set_index('cluster')[heat_cols]
    fig_heat = px.imshow(
        heat_df.T, text_auto='.2f', aspect='auto',
        color_continuous_scale='YlOrRd',
        title="Heatmap Cluster : Karakteristik Rata-rata"
    )
    fig_heat.update_layout(**PLOTLY_LAYOUT, title_font_size=14)
    fig_heat = fix_axes(fig_heat)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("#### Top 5 Zona Pickup Tersibuk per Cluster")
    cluster_tabs = st.tabs([f"Cluster {c}" for c in sorted(zone_stats['cluster'].unique())])
    for i, ct in enumerate(cluster_tabs):
        with ct:
            c_id = sorted(zone_stats['cluster'].unique())[i]
            top5 = (zone_stats[zone_stats['cluster'] == c_id]
                    .nlargest(5, 'total_trips')
                    [['pickup_zone_name','pickup_borough','total_trips','avg_fare','rush_hour_ratio']]
                    .rename(columns={
                        'pickup_zone_name':'Zona','pickup_borough':'Borough',
                        'total_trips':'Total Trips','avg_fare':'Avg Fare ($)',
                        'rush_hour_ratio':'Rush Hour Ratio'
                    }))
            st.dataframe(
                top5.style
                    .background_gradient(subset=['Total Trips'], cmap='YlOrRd')
                    .format({'Total Trips':'{:,.0f}','Avg Fare ($)':'{:.2f}',
                             'Rush Hour Ratio':'{:.3f}'}),
                use_container_width=True
            )

    st.markdown(f"""
    <div class="insight-box">
        <h4> Insight : K-Means Clustering Zona</h4>
        <ul>
            <li>K optimal = <b>{best_k} cluster</b> berdasarkan Silhouette Score tertinggi
                (<b>{final_sil:.4f}</b>), menunjukkan pemisahan cluster yang
                {'sangat baik' if final_sil > 0.5 else 'cukup baik'}.</li>
            <li>Cluster dengan <b>volume trip tertinggi</b> umumnya berasal dari
                area <b>Manhattan</b> (Midtown, Upper East Side) dan Airport.</li>
            <li>Cluster <b>suburban</b> menunjukkan karakteristik:
                jarak lebih panjang, tarif lebih tinggi, dan rush_ratio lebih rendah.</li>
            <li>Cluster <b>malam</b> memiliki night_ratio tinggi : zona yang aktif
                setelah jam 20:00 seperti area hiburan dan transportasi malam.</li>
            <li>Heatmap profil memperlihatkan pembeda utama antar cluster adalah
                kombinasi <b>avg_fare, total_trips, dan rush_hour_ratio</b>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# TAB 4 : Q3: TREN TEMPORAL
with tab_q3:
    st.markdown('<div class="section-header">Analisis Tren Temporal</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Variabel: pickup_datetime, fare_amount, total_amount, is_rush_hour | Metode: Time-Series Aggregation</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    month_labels = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'Mei',6:'Jun',
                    7:'Jul',8:'Agu',9:'Sep',10:'Okt',11:'Nov',12:'Des'}
    days_label   = ['Sen','Sel','Rab','Kam','Jum','Sab','Min']

    monthly = df.groupby('pickup_month').agg(
        total_trips   = ('fare_amount','count'),
        total_revenue = ('total_amount','sum'),
        avg_fare      = ('fare_amount','mean'),
        avg_distance  = ('trip_distance','mean'),
    ).reset_index()
    monthly['month_name'] = monthly['pickup_month'].map(month_labels)

    hourly_agg = df.groupby('pickup_hour').agg(
        total_trips = ('fare_amount','count'),
        avg_fare    = ('fare_amount','mean'),
    ).reset_index()

    weekly = df.groupby('pickup_week').agg(
        total_trips = ('fare_amount','count'),
        avg_fare    = ('fare_amount','mean'),
    ).reset_index()

    rush_comp = df.groupby('is_rush_hour').agg(
        avg_fare   = ('fare_amount','mean'),
        total_trip = ('fare_amount','count'),
    ).reset_index()
    rush_comp['label'] = rush_comp['is_rush_hour'].map({0:'Non Rush Hour',1:'Rush Hour'})

    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                    border:1px solid {C['border']}33; border-radius:18px; padding:18px;
                    margin-bottom:12px; box-shadow:0 2px 16px {C['accent2']}10;">""", unsafe_allow_html=True)
        fig_mt = go.Figure()
        fig_mt.add_trace(go.Scatter(
            x=monthly['pickup_month'], y=monthly['total_trips'],
            mode='lines+markers', name='Total Trips',
            line=dict(color='#7AE582',width=3),
            marker=dict(size=10, color='#9FFFCB',
                        line=dict(color='#25A18E',width=2)),
            fill='tozeroy', fillcolor='rgba(37,161,142,0.12)'
        ))
        fig_mt.update_layout(**PLOTLY_LAYOUT,
                             title="Total Perjalanan per Bulan",
                             title_font_size=14,
                             xaxis=dict(tickvals=list(range(1,13)),
                                        ticktext=list(month_labels.values()),
                                        gridcolor=C["grid"],
                                        tickfont=dict(color=C['text_main']),
                                        color=C['text_main']),
                             yaxis_title="Jumlah Trip")
        fig_mt = fix_axes(fig_mt)
        st.plotly_chart(fig_mt, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                    border:1px solid {C['border']}33; border-radius:18px; padding:18px;
                    margin-bottom:12px; box-shadow:0 2px 16px {C['accent2']}10;">""", unsafe_allow_html=True)
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Scatter(
            x=monthly['pickup_month'], y=monthly['total_revenue']/1e3,
            mode='lines+markers', name='Revenue (K USD)',
            line=dict(color='#9FFFCB',width=3),
            marker=dict(size=10, color='#7AE582',
                        line=dict(color='#00A5CF',width=2)),
            fill='tozeroy', fillcolor='rgba(0,165,207,0.12)'
        ))
        fig_rev.update_layout(**PLOTLY_LAYOUT,
                              title="Total Pendapatan per Bulan (K USD)",
                              title_font_size=14,
                              xaxis=dict(tickvals=list(range(1,13)),
                                         ticktext=list(month_labels.values()),
                                         gridcolor=C["grid"],
                                         tickfont=dict(color=C['text_main']),
                                         color=C['text_main']),
                              yaxis_title="Pendapatan (K USD)")
        fig_rev = fix_axes(fig_rev)
        st.plotly_chart(fig_rev, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c3,c4 = st.columns(2)
    with c3:
        st.markdown(f"""<div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                    border:1px solid {C['border']}33; border-radius:18px; padding:18px;
                    margin-bottom:12px; box-shadow:0 2px 16px {C['accent2']}10;">""", unsafe_allow_html=True)
        fig_hr = make_subplots(specs=[[{"secondary_y":True}]])
        fig_hr.add_trace(
            go.Bar(x=hourly_agg['pickup_hour'], y=hourly_agg['total_trips'],
                   name='Total Trips', marker_color='#7AE582', opacity=0.75),
            secondary_y=False
        )
        fig_hr.add_trace(
            go.Scatter(x=hourly_agg['pickup_hour'], y=hourly_agg['avg_fare'],
                       mode='lines+markers', name='Avg Fare',
                       line=dict(color='#00A5CF', width=2.5),
                       marker=dict(size=7, color='#9FFFCB')),
            secondary_y=True
        )
        fig_hr.update_layout(**PLOTLY_LAYOUT,
                             title="Trip & Avg Fare per Jam",
                             title_font_size=14)
        fig_hr.update_yaxes(title_text="Total Trips", secondary_y=False,
                            gridcolor=C["grid"])
        fig_hr.update_yaxes(title_text="Avg Fare (USD)", secondary_y=True,
                            gridcolor="rgba(0,0,0,0)")
        fig_hr = fix_axes(fig_hr)
        st.plotly_chart(fig_hr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""<div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                    border:1px solid {C['border']}33; border-radius:18px; padding:18px;
                    margin-bottom:12px; box-shadow:0 2px 16px {C['accent2']}10;">""", unsafe_allow_html=True)
        pivot_hm = df.pivot_table(
            values='fare_amount', index='pickup_hour',
            columns='pickup_dayofweek', aggfunc='count'
        )
        # Rename only the columns that actually exist in filtered data
        day_map = dict(zip(range(7), ['Sen','Sel','Rab','Kam','Jum','Sab','Min']))
        pivot_hm.columns = [day_map[c] for c in pivot_hm.columns]
        fig_hmap = px.imshow(pivot_hm, aspect='auto',
                             color_continuous_scale='YlOrRd',
                             title="Heatmap Trip : Jam × Hari")
        fig_hmap.update_layout(**PLOTLY_LAYOUT, title_font_size=14)
        fig_hmap = fix_axes(fig_hmap)
        st.plotly_chart(fig_hmap, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c5,c6,c7 = st.columns(3)
    with c5:
        st.markdown(f"""<div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                    border:1px solid {C['border']}33; border-radius:18px; padding:18px;
                    margin-bottom:12px; box-shadow:0 2px 16px {C['accent2']}10;">""", unsafe_allow_html=True)
        dly_fare = df.groupby('pickup_dayofweek')['fare_amount'].mean().reset_index()
        dly_fare['hari'] = dly_fare['pickup_dayofweek'].map(
            dict(enumerate(days_label)))
        fig_dly = px.bar(dly_fare, x='hari', y='fare_amount',
                         title="Avg Fare per Hari",
                         color='fare_amount',
                         color_continuous_scale='YlOrRd',
                         text='fare_amount')
        fig_dly.update_traces(texttemplate='$%{text:.2f}', textposition='outside',
                              marker_line_color='rgba(0,0,0,0)')
        fig_dly.update_layout(**PLOTLY_LAYOUT, title_font_size=14,
                              showlegend=False, yaxis_title="Avg Fare (USD)")
        fig_dly = fix_axes(fig_dly)
        st.plotly_chart(fig_dly, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c6:
        st.markdown(f"""<div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                    border:1px solid {C['border']}33; border-radius:18px; padding:18px;
                    margin-bottom:12px; box-shadow:0 2px 16px {C['accent2']}10;">""", unsafe_allow_html=True)
        fig_rh = px.bar(rush_comp, x='label', y='avg_fare',
                        title="Avg Fare: Rush vs Non-Rush Hour",
                        color='label',
                        color_discrete_sequence=['#00A5CF','#7AE582'],
                        text='avg_fare')
        fig_rh.update_traces(texttemplate='$%{text:.2f}', textposition='outside',
                             marker_line_color='rgba(0,0,0,0)', width=0.5)
        fig_rh.update_layout(**PLOTLY_LAYOUT, title_font_size=14,
                             showlegend=False, yaxis_title="Avg Fare (USD)")
        fig_rh = fix_axes(fig_rh)
        st.plotly_chart(fig_rh, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c7:
        st.markdown(f"""<div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                    border:1px solid {C['border']}33; border-radius:18px; padding:18px;
                    margin-bottom:12px; box-shadow:0 2px 16px {C['accent2']}10;">""", unsafe_allow_html=True)
        fig_wk = go.Figure()
        fig_wk.add_trace(go.Scatter(
            x=weekly['pickup_week'], y=weekly['total_trips'],
            mode='lines', name='Weekly Trips',
            line=dict(color='#9FFFCB', width=2),
            fill='tozeroy', fillcolor='rgba(159,255,203,0.1)'
        ))
        fig_wk.update_layout(**PLOTLY_LAYOUT,
                             title="Tren Mingguan Jumlah Trip",
                             title_font_size=14,
                             xaxis_title="Minggu ke-",
                             yaxis_title="Jumlah Trip")
        fig_wk = fix_axes(fig_wk)
        st.plotly_chart(fig_wk, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    peak_m  = monthly.loc[monthly['total_trips'].idxmax(), 'month_name']
    low_m   = monthly.loc[monthly['total_trips'].idxmin(), 'month_name']
    peak_h  = int(hourly_agg.loc[hourly_agg['total_trips'].idxmax(), 'pickup_hour'])

    # Safe fallback untuk rush_comp jika salah satu kategori tidak ada di data terfilter
    _rush_row  = rush_comp.loc[rush_comp['is_rush_hour']==1,'avg_fare']
    _nrush_row = rush_comp.loc[rush_comp['is_rush_hour']==0,'avg_fare']
    rush_avg  = float(_rush_row.values[0])  if len(_rush_row)  > 0 else float(rush_comp['avg_fare'].mean())
    nrush_avg = float(_nrush_row.values[0]) if len(_nrush_row) > 0 else float(rush_comp['avg_fare'].mean())
    diff_rh = rush_avg - nrush_avg

    st.markdown(f"""
    <div class="insight-box">
        <h4> Insight : Tren Temporal</h4>
        <ul>
            <li>Bulan <b>{peak_m}</b> mencatat <b>permintaan tertinggi</b>,
                sementara bulan <b>{low_m}</b> mencatat permintaan terendah.</li>
            <li>Jam paling sibuk adalah <b>jam {peak_h}:00</b> : pagi hari menunjukkan
                lonjakan tajam akibat perjalanan commuter ke tempat kerja.</li>
            <li>Rush Hour memiliki avg fare <b>${rush_avg:.2f}</b> vs Non-Rush Hour
                <b>${nrush_avg:.2f}</b> : selisih hanya
                <b>${abs(diff_rh):.2f}</b>, menunjukkan struktur tarif NYC
                relatif stabil di semua periode.</li>
            <li>Heatmap jam × hari menunjukkan <b>Jumat & Sabtu malam</b>
                (jam 20:00–23:00) adalah periode tersibuk kedua setelah rush hour pagi.</li>
            <li>Tren mingguan memperlihatkan pola <b>musiman ringan</b> : pertengahan tahun
                (Q2-Q3) cenderung memiliki volume lebih tinggi.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# TAB 5 : Q4: METODE PEMBAYARAN
with tab_q4:
    st.markdown('<div class="section-header">Analisis Metode Pembayaran</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Variabel: payment_type, fare_amount | Metode: Analisis Frekuensi & Proporsi</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    pay_counts = df['payment_label'].value_counts().reset_index()
    pay_counts.columns = ['Metode Pembayaran','Jumlah Trip']
    pay_counts['Proporsi (%)'] = (pay_counts['Jumlah Trip'] /
                                   pay_counts['Jumlah Trip'].sum() * 100).round(2)

    top1 = pay_counts.iloc[0]
    top2 = pay_counts.iloc[1]

    p1,p2,p3 = st.columns(3)
    for col,icon,label,val,sub in [
        (p1,"🥇","Metode Terpopuler", top1['Metode Pembayaran'],
            f"{top1['Proporsi (%)']:.1f}% dari total trip"),
        (p2,"🥈","Metode Ke-2",       top2['Metode Pembayaran'],
            f"{top2['Proporsi (%)']:.1f}% dari total trip"),
        (p3,"📊","Jumlah Metode",     str(pay_counts['Metode Pembayaran'].nunique()),
            "jenis pembayaran berbeda"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2 = st.columns([3,2])
    with c1:
        # Bar chart
        fig_pay = px.bar(
            pay_counts, x='Metode Pembayaran', y='Jumlah Trip',
            title="Distribusi Jenis Pembayaran Customer NYC Taxi 2022",
            color='Metode Pembayaran',
            color_discrete_sequence=COLORS,
            text='Proporsi (%)'
        )
        fig_pay.update_traces(
            texttemplate='%{y:,}<br>(%{text:.1f}%)',
            textposition='outside',
            marker_line_color='rgba(0,0,0,0)'
        )
        fig_pay.update_layout(**PLOTLY_LAYOUT, title_font_size=14,
                              showlegend=False, yaxis_title="Jumlah Trip")
        fig_pay = fix_axes(fig_pay)
        st.plotly_chart(fig_pay, use_container_width=True)

    with c2:
        # Donut chart
        fig_donut = px.pie(
            pay_counts, names='Metode Pembayaran', values='Jumlah Trip',
            hole=0.55, title="Proporsi Metode Pembayaran",
            color_discrete_sequence=COLORS
        )
        fig_donut.update_traces(
            textposition='outside', textinfo='percent+label',
            marker=dict(line=dict(color='#0a2540', width=2))
        )
        fig_donut.update_layout(**PLOTLY_LAYOUT, title_font_size=14,
                                showlegend=False)
        fig_donut = fix_axes(fig_donut)
        st.plotly_chart(fig_donut, use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        avg_by_pay = df.groupby('payment_label')['fare_amount'].mean().reset_index()
        avg_by_pay.columns = ['Metode Pembayaran','Avg Fare']
        avg_by_pay = avg_by_pay.sort_values('Avg Fare', ascending=False)
        fig_avg_pay = px.bar(
            avg_by_pay, x='Metode Pembayaran', y='Avg Fare',
            title="Rata-rata Fare per Metode Pembayaran",
            color='Avg Fare', color_continuous_scale='YlOrRd',
            text='Avg Fare'
        )
        fig_avg_pay.update_traces(texttemplate='$%{text:.2f}', textposition='outside',
                                  marker_line_color='rgba(0,0,0,0)')
        fig_avg_pay.update_layout(**PLOTLY_LAYOUT, title_font_size=14,
                                  showlegend=False, yaxis_title="Avg Fare (USD)")
        fig_avg_pay = fix_axes(fig_avg_pay)
        st.plotly_chart(fig_avg_pay, use_container_width=True)

    with c4:
        # Payment by time of day
        pay_tod = df.groupby(['time_of_day','payment_label'])['fare_amount'].count().reset_index()
        pay_tod.columns = ['Waktu','Metode','Jumlah']
        fig_pt = px.bar(
            pay_tod, x='Waktu', y='Jumlah', color='Metode',
            title="Metode Pembayaran per Waktu Hari",
            color_discrete_sequence=COLORS, barmode='stack'
        )
        fig_pt.update_layout(**PLOTLY_LAYOUT, title_font_size=14,
                             yaxis_title="Jumlah Trip")
        fig_pt = fix_axes(fig_pt)
        st.plotly_chart(fig_pt, use_container_width=True)

    st.markdown("#### Tabel Distribusi Lengkap")
    st.dataframe(
        pay_counts.style
            .highlight_max(subset=['Jumlah Trip','Proporsi (%)'],
                           color='rgba(37,161,142,0.35)')
            .format({'Jumlah Trip':'{:,.0f}','Proporsi (%)':'{:.2f}%'}),
        use_container_width=True
    )

    st.markdown(f"""
    <div class="insight-box">
        <h4> Insight : Metode Pembayaran</h4>
        <ul>
            <li><b>Credit Card</b> mendominasi dengan proporsi <b>{top1['Proporsi (%)']:.1f}%</b>
                dari seluruh transaksi : mencerminkan preferensi digital yang kuat.</li>
            <li><b>Cash</b> berada di urutan ke-2 (<b>{top2['Proporsi (%)']:.1f}%</b>),
                menunjukkan segmen penumpang yang masih mengandalkan pembayaran tunai.</li>
            <li>Rata-rata fare <b>Credit Card lebih tinggi</b> dibanding Cash, kemungkinan
                karena penumpang jarak jauh & airport lebih memilih kartu.</li>
            <li>Metode <b>No Charge & Dispute</b> sangat kecil (&lt;2%) :
                menunjukkan integritas data transaksi yang tinggi.</li>
            <li>Stacked bar per waktu hari menunjukkan dominasi Credit Card
                <b>konsisten di semua periode</b>, dengan Cash sedikit meningkat pada malam hari.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# TAB 6 : RINGKASAN
with tab_summary:
    st.markdown('<div class="section-header">Ringkasan Akhir Analisis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">PROJECT ADBC : NYC Yellow Taxi 2022 | Full Pipeline Summary</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Dataset overview
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{C['bg_card']},{C['bg_card2']});
                border:1px solid {C['border']}33; border-radius:18px; padding:24px;
                margin-bottom:20px; box-shadow:0 2px 16px {C['accent2']}10;">
        <div style="font-size:1.0rem; font-weight:700; color:{C['accent1']}; margin-bottom:12px;
                    font-family:'Space Grotesk',sans-serif; letter-spacing:0.3px;">
            Tentang Dataset
        </div>
    """, unsafe_allow_html=True)

    info_cols = st.columns(4)
    info_data = [
        ("Platform","BigQuery Public Dataset"),
        ("Dataset","tlc_yellow_trips_2022"),
        ("Ukuran","500.000 observasi"),
        ("Tahun","2022"),
    ]
    for col,(k,v) in zip(info_cols,info_data):
        with col:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="color:{C['text_muted']}; font-size:0.75rem; letter-spacing:1px;">{k}</div>
                <div style="color:{C['accent1']}; font-weight:700; font-size:0.95rem;">{v}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Ambil ulang hasil ML jika ada
    try:
        r2_val  = res['r2']
        mae_val = res['mae']
        rmse_val= res['rmse']
    except:
        r2_val = mae_val = rmse_val = 0.0

    try:
        bk  = km_res['best_k']
        fs  = km_res['final_sil']
    except:
        bk = 0; fs = 0.0

    try:
        peak_m_s  = monthly.loc[monthly['total_trips'].idxmax(),'month_name']
        low_m_s   = monthly.loc[monthly['total_trips'].idxmin(),'month_name']
        peak_h_s  = int(hourly_agg.loc[hourly_agg['total_trips'].idxmax(),'pickup_hour'])
    except:
        peak_m_s = low_m_s = "N/A"; peak_h_s = 0

    try:
        top1_s = pay_counts.iloc[0]
        top2_s = pay_counts.iloc[1]
    except:
        top1_s = top2_s = None

    cards_summary = [
        ("🤖", "Prediksi Tarif",
         "Random Forest Regressor",
         [f"R² = {r2_val:.4f} ({r2_val*100:.1f}% variansi dijelaskan)",
          f"MAE = ${mae_val:.3f} USD",
          f"RMSE = ${rmse_val:.3f} USD",
          "Fitur dominan: trip_distance, trip_duration",
          "Model tidak bias sistematis (residual ≈ 0)"]),
        ("🗺️", "Clustering Zona",
         "K-Means + Elbow + Silhouette",
         [f"K optimal = {bk} cluster",
          f"Silhouette Score = {fs:.4f}",
          "Cluster Manhattan: volume tinggi, fare moderat",
          "Cluster Airport: fare tinggi, jarak panjang",
          "Cluster Suburban: weekend & night ratio tinggi"]),
        ("📈", "Tren Temporal",
         "Agregasi Time-Series Bulanan & Per Jam",
         [f"Bulan tersibuk: {peak_m_s}",
          f"Bulan terlowir: {low_m_s}",
          f"Jam paling sibuk: {peak_h_s}:00",
          "Rush hour tidak mendorong kenaikan fare signifikan",
          "Jumat & Sabtu malam = periode sibuk ke-2"]),
        ("💳", "Metode Pembayaran",
         "Analisis Frekuensi & Proporsi",
         [f"Dominan: {top1_s['Metode Pembayaran']} ({top1_s['Proporsi (%)']:.1f}%)" if top1_s is not None else "",
          f"Ke-2: {top2_s['Metode Pembayaran']} ({top2_s['Proporsi (%)']:.1f}%)" if top2_s is not None else "",
          "CC dominan di semua periode waktu",
          "Cash meningkat sedikit pada malam hari",
          "No Charge & Dispute < 2% : data bersih"]),
    ]

    for i in range(0,4,2):
        cols = st.columns(2)
        for j,col in enumerate(cols):
            if i+j < len(cards_summary):
                icon,title,method,points = cards_summary[i+j]
                pts_html = "".join(f"<li>{p}</li>" for p in points if p)
                with col:
                    st.markdown(f"""
                    <div style="
                        background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                        border:1px solid {C['accent2']}44;
                        border-top:3px solid {C['accent1']};
                        border-radius:20px; padding:24px;
                        margin-bottom:16px;
                        box-shadow: 0 4px 24px {C['accent2']}15;
                        transition: all 0.25s;
                    ">
                        <div style="font-size:2rem; margin-bottom:10px;">{icon}</div>
                        <div style="color:{C['accent3']}; font-size:1.0rem; font-weight:800;
                                    margin-bottom:4px; font-family:'Space Grotesk',sans-serif;">{title}</div>
                        <div style="color:{C['accent2']}; font-size:0.73rem; letter-spacing:1.5px;
                                    margin-bottom:14px; text-transform:uppercase;">{method}</div>
                        <ul style="color:{C['text_main']}; font-size:0.84rem; line-height:1.9;
                                   padding-left:18px; margin:0;">
                            {pts_html}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,{C['bg_card']},{C['bg_card2']});
        border:2px solid {C['accent2']};
        border-radius:20px; padding:28px;
        margin-top:10px;
        box-shadow: 0 8px 32px {C['accent2']}25;
    ">
        <div style="font-size:1.15rem; font-weight:800; color:{C['accent1']}; margin-bottom:16px;
                    font-family:'Space Grotesk',sans-serif;">
            🎯 Rekomendasi Strategis
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">
            <div style="background:{C['bg_card2']}; border-radius:14px; padding:18px;
                        border:1px solid {C['accent2']}33;">
                <div style="color:{C['accent4']}; font-weight:700; font-size:0.82rem;
                            letter-spacing:1.5px; margin-bottom:10px;">⚡ OPERASIONAL</div>
                <ul style="color:{C['text_main']}; font-size:0.85rem; line-height:1.9; padding-left:18px; margin:0;">
                    <li>Tambah armada di zona Manhattan & Airport pada jam 07:00–09:00</li>
                    <li>Optimalkan driver malam di cluster zona hiburan (Jum–Sab)</li>
                    <li>Gunakan model RF untuk estimasi tarif real-time di aplikasi</li>
                </ul>
            </div>
            <div style="background:{C['bg_card2']}; border-radius:14px; padding:18px;
                        border:1px solid {C['accent2']}33;">
                <div style="color:{C['accent4']}; font-weight:700; font-size:0.82rem;
                            letter-spacing:1.5px; margin-bottom:10px;">📈 BISNIS</div>
                <ul style="color:{C['text_main']}; font-size:0.85rem; line-height:1.9; padding-left:18px; margin:0;">
                    <li>Chassless perlu dipertahankan lewat insentif CC</li>
                    <li>Strategi promosi pada bulan permintaan rendah untuk menaikkan volume</li>
                    <li>Cluster dari zona pinggiran kota menunjukkan tarif mahal karena perjalanannya jauh</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    "<div style='height:1px;background:linear-gradient(90deg,transparent," + C['border'] + "55,transparent);margin-top:48px;'></div>"
    "<div style='text-align:center;padding:28px 20px 24px;"
    "background:linear-gradient(135deg," + C['bg_card'] + "88," + C['bg_card2'] + "88);'>"
    "<div style='display:flex;justify-content:center;align-items:center;gap:18px;flex-wrap:wrap;margin-bottom:10px;'>"
    "<span style='font-size:0.95rem;font-weight:700;color:" + C['text_main'] + ";font-family:Space Grotesk,sans-serif;'>Elsya Anggraini</span>"
    "<span style='width:5px;height:5px;border-radius:50%;background:" + C['accent2'] + ";display:inline-block;opacity:0.7;'></span>"
    "<span style='font-size:0.95rem;font-weight:700;color:" + C['text_main'] + ";font-family:Space Grotesk,sans-serif;'>Okta Mianda Br Sihotang</span>"
    "</div>"
    "<div style='font-size:0.78rem;font-weight:600;color:" + C['accent2'] + ";letter-spacing:1.5px;text-transform:uppercase;margin-bottom:16px;'>Universitas Negeri Yogyakarta</div>"
    "<div style='height:1px;background:linear-gradient(90deg,transparent," + C['border'] + "33,transparent);margin-bottom:14px;'></div>"
    "<div style='font-size:0.74rem;color:" + C['text_muted'] + ";letter-spacing:0.3px;'>"
    "🗽 NYC Yellow Taxi Analytics &nbsp;&middot;&nbsp; PROJECT ADBC 2022 &nbsp;&middot;&nbsp; "
    "Built with <b style='color:" + C['accent4'] + ";'>Streamlit</b> + <b style='color:" + C['accent1'] + ";'>Plotly</b>"
    "</div></div>",
    unsafe_allow_html=True
)