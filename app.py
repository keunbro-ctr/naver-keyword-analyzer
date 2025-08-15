import streamlit as st
import os, time, hmac, hashlib, base64, requests, json, math
import pandas as pd
import plotly.graph_objects as go
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from typing import List, Dict
import io

# ========================================
# 페이지 & 공통 스타일
# ========================================
st.set_page_config(
    page_title="키워드 검색량 추이 분석기",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
    }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px; }
    .grade-card { padding: 12px; border-radius: 8px; text-align: center; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .grade-a-plus { background: linear-gradient(135deg, #28a745, #20c997); color: #fff !important; border: 2px solid #28a745; }
    .grade-a { background: linear-gradient(135deg, #5cb85c, #6bc97f); color: #fff !important; border: 2px solid #5cb85c; }
    .grade-b-plus { background: linear-gradient(135deg, #ffc107, #ffcd38); color: #212529 !important; border: 2px solid #ffc107; }
    .grade-b { background: linear-gradient(135deg, #ff9800, #ffa726); color: #fff !important; border: 2px solid #ff9800; }
    .grade-c { background: linear-gradient(135deg, #dc3545, #e85d6c); color: #fff !important; border: 2px solid #dc3545; }
    .stSlider > div > div > div > div {
        background-image: linear-gradient(to right, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# 세션 상태
# ========================================
# ========================================
# 세션 상태 초기화 수정
# ========================================
if 'api_keys_saved' not in st.session_state:
    st.session_state.api_keys_saved = False

# API 키 저장소 초기화
for k in ["datalab_client_id","datalab_client_secret","ads_api_key","ads_secret","ads_customer_id"]:
    if k not in st.session_state:
        st.session_state[k] = ""

# 입력 위젯 키 초기화
input_keys = ["datalab_id_input", "datalab_secret_input", "ads_key_input", "ads_secret_input", "ads_cid_input"]
for k in input_keys:
    if k not in st.session_state:
        st.session_state[k] = ""

# 라디오 기본값 설정
if "api_input_method" not in st.session_state:
    st.session_state["api_input_method"] = "개별 입력"

# 일괄 붙여넣기 텍스트 초기화
if "bulk_text" not in st.session_state:
    st.session_state["bulk_text"] = ""

KEY_MAP = {
    "datalab_client_id": "datalab_id_input",
    "datalab_client_secret": "datalab_secret_input",
    "ads_api_key": "ads_key_input",
    "ads_secret": "ads_secret_input",
    "ads_customer_id": "ads_cid_input",
}

def sync_saved_to_inputs(force: bool = False):
    """저장된 API 키를 입력 위젯에 동기화"""
    for saved_key, input_key in KEY_MAP.items():
        saved_val = st.session_state.get(saved_key, "")
        if force or not st.session_state.get(input_key):
            st.session_state[input_key] = saved_val

def validate_api_keys():
    """API 키 유효성 검사"""
    required_keys = ["datalab_client_id", "datalab_client_secret", "ads_api_key", "ads_secret", "ads_customer_id"]
    
    for key in required_keys:
        value = st.session_state.get(key, "")
        if not value or not value.strip():
            return False, f"{key} 값이 비어있습니다."
    
    return True, "모든 API 키가 입력되었습니다."

# ========================================
# 유틸리티
# ========================================
def last_day_of_prev_month(today=None) -> str:
    today = today or date.today()
    first_this_month = date(today.year, today.month, 1)
    last_prev = first_this_month - relativedelta(days=1)
    return last_prev.strftime("%Y-%m-%d")

def _to_int(x):
    try: 
        return int(x)
    except:
        try: 
            return int(float(x))
        except: 
            return 0

def _normalize(s: str) -> str:
    """소문자 + 모든 공백 제거"""
    return "".join(s.lower().split())

def clean_keyword(keyword: str) -> str:
    """공백 제거(모든 종류), 양끝 trim"""
    return "".join(keyword.strip().split())

def unique_cleaned_list(lines: list[str]) -> tuple[list[str], list[str]]:
    """공백 제거 후 중복 제거(순서 보존). (정상목록, 제거된중복목록) 반환"""
    seen = set()
    kept, dropped = [], []
    for ln in lines:
        if not ln.strip():
            continue
        ck = clean_keyword(ln)
        if ck in seen:
            dropped.append(ck)
            continue
        seen.add(ck)
        kept.append(ck)
    return kept, dropped

def parse_bulk_api_keys(text: str) -> dict:
    """
    일괄 붙여넣기 파서 - 수정된 버전
    """
    result = {}
    
    if not text or not text.strip():
        return result
    
    # 텍스트 전처리: 다양한 구분자로 분리 시도
    text = text.strip()
    
    # 줄바꿈이 있는 경우
    if '\n' in text:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # 줄바꿈이 없지만 공백이나 탭으로 구분된 경우  
    elif '\t' in text or '  ' in text:
        # 탭이나 2개 이상의 공백으로 분리
        import re
        lines = [ln.strip() for ln in re.split(r'\s{2,}|\t', text) if ln.strip()]
    # 쉼표로 구분된 경우
    elif ',' in text:
        lines = [ln.strip() for ln in text.split(',') if ln.strip()]
    # 세미콜론으로 구분된 경우
    elif ';' in text:
        lines = [ln.strip() for ln in text.split(';') if ln.strip()]
    # 하나의 긴 문자열인 경우 - 길이로 추정해서 분리
    else:
        # 전체 길이가 100자 이상이면 5등분 시도
        if len(text) > 100:
            chunk_size = len(text) // 5
            lines = []
            for i in range(5):
                start = i * chunk_size
                if i == 4:  # 마지막 청크는 끝까지
                    chunk = text[start:].strip()
                else:
                    chunk = text[start:start + chunk_size].strip()
                if chunk:
                    lines.append(chunk)
        else:
            lines = [text.strip()]
    
    print(f"DEBUG: 분리된 라인들: {lines}")
    print(f"DEBUG: 라인 개수: {len(lines)}")
    
    # key=value 형식인지 확인
    # key=value 형식인지 확인 (더 정확한 감지)
    # API 키 끝의 ==는 제외하고, 실제 key=value 패턴만 감지
    has_equals = any(
        '=' in ln and 
        not ln.endswith('=') and 
        not ln.endswith('==') and
        ln.count('=') == 1 and
        len(ln.split('=')[0].strip()) > 0 and
        len(ln.split('=')[1].strip()) > 0
        for ln in lines
    )
    
    if has_equals:
        # key=value 형식 처리
        for ln in lines:
            if '=' in ln:
                parts = ln.split('=', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip()
                    if value:
                        result[key] = value
    else:
        # 순서대로 입력된 경우
        keys_in_order = [
            "datalab_client_id", 
            "datalab_client_secret", 
            "ads_api_key", 
            "ads_secret", 
            "ads_customer_id"
        ]
        
        for i, key in enumerate(keys_in_order):
            if i < len(lines):
                value = lines[i].strip()
                if value:
                    result[key] = value
                    print(f"DEBUG: {key} = {value}")
    
    # 키 별칭 처리 (기존과 동일)
    alias = {
        "client_id": "datalab_client_id",
        "client secret": "datalab_client_secret", 
        "client_secret": "datalab_client_secret",
        "datalab_id": "datalab_client_id",
        "datalab_secret": "datalab_client_secret",
        "api_key": "ads_api_key",
        "ads_key": "ads_api_key", 
        "secret": "ads_secret",
        "ads_secret": "ads_secret",
        "customer_id": "ads_customer_id",
        "ads_customer_id": "ads_customer_id",
        "cid": "ads_customer_id"
    }
    
    final_result = {}
    for key, value in result.items():
        normalized_key = alias.get(key, key)
        if value and value.strip():
            final_result[normalized_key] = value.strip()
    
    print(f"DEBUG: 최종 결과: {final_result}")
    return final_result

# ========================================
# 네이버 광고 API
# ========================================
def sign(ts_ms: str, method: str, uri: str, secret: str) -> str:
    msg = f"{ts_ms}.{method}.{uri}"
    dig = hmac.new(secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).digest()
    return base64.b64encode(dig).decode("utf-8")

def ads_get(path: str, params: dict, api_key: str, secret: str, customer_id: str) -> dict:
    if not path.startswith("/"): path = "/" + path
    ts = str(int(time.time() * 1000))
    headers = {
        "X-Timestamp": ts,
        "X-API-KEY": api_key,
        "X-Customer": str(customer_id),
        "X-Signature": sign(ts, "GET", path, secret),
    }
    r = requests.get("https://api.searchad.naver.com" + path, headers=headers, params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"[ADS] {r.status_code}: {r.text}")
    return r.json()

@st.cache_data(ttl=3600)
def get_absolute_volumes(keywords: List[str], api_key: str, secret: str, customer_id: str) -> Dict[str, int]:
    """키워드들의 최근 1개월 절대 검색량"""
    volumes = {}
    BATCH_SIZE = 5

    cleaned_keywords = [clean_keyword(kw) for kw in keywords]
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, len(cleaned_keywords), BATCH_SIZE):
        batch = cleaned_keywords[i:i+BATCH_SIZE]
        params = {"hintKeywords": ",".join(batch), "showDetail": 1}
        try:
            data = ads_get("/keywordstool", params, api_key, secret, customer_id)
            items = data.get("keywordList", [])
            by_rel = {_normalize(it.get("relKeyword","")): it for it in items}
            for idx, kw in enumerate(batch):
                original_kw = cleaned_keywords[i + idx]  # 표시도 공백 제거 버전 사용
                it = by_rel.get(_normalize(kw))
                if not it:
                    it = next((v for k,v in by_rel.items() if _normalize(kw) in k or k in _normalize(kw)), None)
                    if not it:
                        volumes[original_kw] = 0
                        continue
                pc = _to_int(it.get("monthlyPcQcCnt", 0))
                mobile = _to_int(it.get("monthlyMobileQcCnt", 0))
                volumes[original_kw] = pc + mobile
        except Exception as e:
            st.warning(f"⚠️ 배치 처리 오류: {batch} - {e}")
            for idx, kw in enumerate(batch):
                original_kw = cleaned_keywords[i + idx]
                volumes[original_kw] = 0
        progress = min((i + BATCH_SIZE) / max(len(cleaned_keywords),1), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"처리 중: {min(i + BATCH_SIZE, len(cleaned_keywords))}/{len(cleaned_keywords)} 키워드")
        time.sleep(0.05)

    progress_bar.empty()
    status_text.empty()
    return volumes

# ========================================
# 네이버 데이터랩 API (최대 10년, 60개월 창+정렬)
# ========================================
def call_datalab(payload, client_id: str, client_secret: str):
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
        "Content-Type": "application/json; charset=UTF-8",
    }
    r = requests.post("https://openapi.naver.com/v1/datalab/search",
                      headers=headers,
                      data=json.dumps(payload).encode("utf-8"),
                      timeout=20)
    if r.status_code != 200:
        raise SystemExit(f"[DataLab ERROR] {r.status_code} {r.text}")
    return r.json()

def month_windows_for(total_months: int, end_date: date, window_size: int = 60):
    """올드→뉴 순서로, 1개월 오버랩을 포함한 (start,end) 윈도우 생성"""
    start_all = (end_date - relativedelta(months=total_months-1)).replace(day=1)
    windows = []
    ws = start_all
    while True:
        we = min(ws + relativedelta(months=window_size-1), end_date)
        windows.append((ws, we))
        if we == end_date:
            break
        # 다음 창 시작 = 바로 이전 창의 종료(1개월 오버랩)
        ws = we
    return windows  # 올드→뉴

@st.cache_data(ttl=3600)
def get_trend_data(keywords: List[str], months: int, client_id: str, client_secret: str) -> pd.DataFrame:
    """데이터랩 상대지수(월) - 최대 10년. 60개월 창으로 나눠 스케일 정렬 후 병합"""
    end_date_str = last_day_of_prev_month()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    cleaned_keywords = [clean_keyword(kw) for kw in keywords]
    rows_all = []

    # 60개월 단위 창 목록
    windows = month_windows_for(months, end_date, window_size=60)

    progress_bar = st.progress(0)
    status_text = st.empty()

    # 창별로 수집
    window_results = {kw: [] for kw in cleaned_keywords}  # kw별 chunk DataFrame 리스트
    BATCH_SIZE = 5

    for w_idx, (ws, we) in enumerate(windows):
        start_date = ws.strftime("%Y-%m-%d")
        end_date_ = we.strftime("%Y-%m-%d")

        for i in range(0, len(cleaned_keywords), BATCH_SIZE):
            batch = cleaned_keywords[i:i+BATCH_SIZE]

            payload = {
                "startDate": start_date,
                "endDate": end_date_,
                "timeUnit": "month",
                "keywordGroups": [{"groupName": kw, "keywords": [kw]} for kw in batch],
            }
            try:
                data = call_datalab(payload, client_id, client_secret)
                for res in data.get("results", []):
                    kw = res.get("title")
                    # 이 창의 df
                    chunk_rows = []
                    for d in res.get("data", []):
                        chunk_rows.append({"keyword": kw, "period": d["period"], "ratio": d["ratio"]})
                    chunk_df = pd.DataFrame(chunk_rows)
                    window_results[kw].append(chunk_df)
            except Exception as e:
                st.warning(f"⚠️ 데이터랩 오류({start_date}~{end_date_}, {batch}): {e}")

        progress = (w_idx+1)/len(windows)
        progress_bar.progress(progress)
        status_text.text(f"추이 데이터 수집: {w_idx+1}/{len(windows)} 창")

    progress_bar.empty()
    status_text.empty()

    # 창 병합(스케일 정렬)
    merged_frames = []
    for kw, chunks in window_results.items():
        if not chunks:
            continue
        # 올드→뉴 순서 chunks
        merged = chunks[0].copy()
        for j in range(1, len(chunks)):
            curr = chunks[j].copy()
            if merged.empty:
                merged = curr
                continue
            if curr.empty:
                continue
            # 겹치는 첫 달(현재 chunk의 첫 period)을 기준으로 스케일 정렬
            overlap_period = curr.iloc[0]["period"]
            prev_overlap = merged[merged["period"] == overlap_period]["ratio"]
            curr_overlap = curr[curr["period"] == overlap_period]["ratio"]
            if not prev_overlap.empty and not curr_overlap.empty and prev_overlap.values[0] > 0:
                scale = curr_overlap.values[0] / prev_overlap.values[0]
                merged["ratio"] = merged["ratio"] * scale
            # 중복되는 overlap row는 현재 chunk 쪽을 보존 -> 이전 병합본에서 제거
            merged = pd.concat([merged[merged["period"] != overlap_period], curr], ignore_index=True)

        merged["keyword"] = kw
        merged_frames.append(merged)

    if not merged_frames:
        raise ValueError("데이터랩에서 유효한 데이터를 가져올 수 없습니다.")

    df = pd.concat(merged_frames, ignore_index=True).sort_values(["keyword", "period"]).reset_index(drop=True)
    return df

def convert_to_absolute_values(trend_df: pd.DataFrame, absolute_volumes: Dict[str, int]) -> pd.DataFrame:
    """상대지수 → 절대검색량"""
    result_df = trend_df.copy()
    result_df["absolute_volume"] = 0

    for keyword in trend_df["keyword"].unique():
        if keyword not in absolute_volumes or absolute_volumes[keyword] == 0:
            continue
        keyword_data = trend_df[trend_df["keyword"] == keyword].copy()
        anchor_abs = absolute_volumes[keyword]

        latest_ratio = keyword_data.iloc[-1]["ratio"]
        if latest_ratio == 0:
            non_zero = keyword_data[keyword_data["ratio"] > 0]
            if non_zero.empty:
                continue
            latest_ratio = non_zero.iloc[-1]["ratio"]

        keyword_data["absolute_volume"] = keyword_data["ratio"].apply(
            lambda r: int(round(anchor_abs * (r / latest_ratio))) if latest_ratio > 0 else 0
        )
        mask = result_df["keyword"] == keyword
        result_df.loc[mask, "absolute_volume"] = keyword_data["absolute_volume"].values

    result_df["date"] = pd.to_datetime(result_df["period"])
    return result_df

# ========================================
# 시각화
# ========================================
def create_interactive_chart(df: pd.DataFrame, hover_mode: str = "closest"):
    """인터랙티브 추이 차트 + 가시성 개선(스파이크/툴팁)"""
    valid_df = df[df["absolute_volume"] > 0].copy()
    if valid_df.empty:
        st.warning("⚠️ 차트로 표시할 유효한 데이터가 없습니다.")
        return None

    fig = go.Figure()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
              '#9B59B6', '#3498DB', '#E74C3C', '#1ABC9C', '#F39C12']

    for i, keyword in enumerate(valid_df["keyword"].unique()):
        kd = valid_df[valid_df["keyword"] == keyword].sort_values("date")
        fig.add_trace(go.Scatter(
            x=kd["date"],
            y=kd["absolute_volume"],
            mode='lines+markers',
            name=keyword,
            line=dict(width=2.6, color=colors[i % len(colors)]),
            marker=dict(size=7, symbol='circle'),
            opacity=0.75,
            hovertemplate=(
                '<b style="font-size: 16px;">%{fullData.name}</b><br>' +
                '<span style="color: #666;">날짜:</span> %{x|%Y년 %m월}<br>' +
                '<span style="color: #666;">검색량:</span> <b style="font-size: 14px;">%{y:,.0f}회</b><br>' +
                '<extra></extra>'
            ),
            hoverlabel=dict(bgcolor='white', font=dict(size=13, color='black'))
        ))

    fig.update_layout(
        title=dict(text='📊 키워드별 월간 검색량 추이', font=dict(size=24, family='Arial Black'), x=0.5, xanchor='center'),
        xaxis=dict(title="기간", gridcolor='rgba(200,200,200,0.3)', showgrid=True,
                   showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=2),
        yaxis=dict(title="월간 검색량 (회)", gridcolor='rgba(200,200,200,0.3)', showgrid=True),
        hovermode=('x unified' if hover_mode == 'x unified' else 'closest'),
        hoverdistance=20,
        template='plotly_white',
        height=550,
        legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
            bgcolor='#11331e', bordercolor='rgba(200,200,200,0.5)', borderwidth=1,
            font=dict(size=12),
            itemclick="toggleothers", itemdoubleclick="toggle"
        ),
        margin=dict(l=80, r=150, t=80, b=80),
    )
    fig.update_yaxes(tickformat=',', tickfont=dict(size=11))
    fig.update_xaxes(tickfont=dict(size=11))
    return fig

def apply_focus(fig: go.Figure, focus_keyword: str | None):
    """선택 키워드만 강조(라인 두껍게, 나머지 투명도↓)"""
    if not fig or not focus_keyword or focus_keyword == "(없음)":
        return fig
    for tr in fig.data:
        if tr.name == focus_keyword:
            tr.opacity = 1.0
            tr.line.width = 3.8
            tr.marker.size = 8
        else:
            tr.opacity = 0.18
            tr.line.width = 1.2
            tr.marker.size = 5
    return fig

# ========================================
# 추천 점수
# ========================================
def calculate_recommendation_score(df: pd.DataFrame, absolute_volumes: Dict[str, int], months: int) -> pd.DataFrame:
    recommendations = []
    for keyword in df["keyword"].unique():
        kd = df[df["keyword"] == keyword].sort_values("date")
        if len(kd) < 2:
            continue

        current_volume = absolute_volumes.get(keyword, 0)
        first_vol = kd.iloc[0]["absolute_volume"]
        last_vol  = kd.iloc[-1]["absolute_volume"]

        # 1) 검색량 점수(40)
        if   current_volume >= 10000: volume_score = 40
        elif current_volume >=  5000: volume_score = 30
        elif current_volume >=  1000: volume_score = 20
        else:                         volume_score = 10

        # 2) 성장률 점수(25)
        growth_rate = 0
        if first_vol > 0:
            growth_rate = ((last_vol - first_vol) / first_vol) * 100
            if   growth_rate >= 50: growth_score = 25
            elif growth_rate >= 20: growth_score = 20
            elif growth_rate >=  0: growth_score = 15
            elif growth_rate >= -20: growth_score = 10
            else:                   growth_score = 5
        else:
            growth_score = 10

        # 3) 안정성 점수(20) - 변동계수
        vols = kd["absolute_volume"].values
        if len(vols) > 1 and vols.mean() > 0:
            cv = vols.std() / vols.mean()
            if   cv <= 0.3: stability_score = 20
            elif cv <= 0.5: stability_score = 15
            elif cv <= 0.8: stability_score = 10
            else:           stability_score = 5
        else:
            cv = 0
            stability_score = 10

        # 4) 트렌드 점수(15) - 최근6개월 vs 직전6개월
        trend_score = 8
        recent_6m_avg = current_volume
        if len(kd) >= 12:
            recent_6m_avg   = kd.tail(6)["absolute_volume"].mean()
            previous_6m_avg = kd.iloc[-12:-6]["absolute_volume"].mean()
            if previous_6m_avg > 0:
                recent_trend = ((recent_6m_avg - previous_6m_avg) / previous_6m_avg) * 100
                if   recent_trend >= 20: trend_score = 15
                elif recent_trend >= 10: trend_score = 12
                elif recent_trend >=  0: trend_score = 10
                elif recent_trend >= -10: trend_score = 7
                else:                    trend_score = 5

        total_score = volume_score + growth_score + stability_score + trend_score
        if   total_score >= 85: grade = "A+ (강력추천)"
        elif total_score >= 75: grade = "A (추천)"
        elif total_score >= 65: grade = "B+ (고려)"
        elif total_score >= 55: grade = "B (보통)"
        else:                   grade = "C (신중)"

        recommendations.append({
            "키워드": keyword,
            "현재검색량": current_volume,
            "성장률(%)": round(growth_rate, 1),
            "검색량점수": volume_score,
            "성장률점수": growth_score,
            "안정성점수": stability_score,
            "트렌드점수": trend_score,
            "총점": total_score,
            "등급": grade,
            "변동계수": round(float(cv), 2),
            "최근6개월평균": round(float(recent_6m_avg), 0)
        })

    rec_df = pd.DataFrame(recommendations)
    if rec_df.empty:
        return rec_df
    rec_df = rec_df.sort_values("총점", ascending=False).reset_index(drop=True)
    rec_df["순위"] = range(1, len(rec_df) + 1)
    return rec_df

def get_grade_class(grade):
    if 'A+' in grade: return 'grade-a-plus'
    if 'A ' in grade or 'A (' in grade: return 'grade-a'
    if 'B+' in grade: return 'grade-b-plus'
    if 'B ' in grade or 'B (' in grade: return 'grade-b'
    return 'grade-c'

# ========================================
# 메인
# ========================================
def save_api_keys():
    st.session_state.api_keys_saved = True

def main():
    st.title("🚀 키워드 검색량 추이 분석기")
    st.markdown("제품 개발을 위한 원료 키워드 트렌드 분석 도구")

# ---------------- Sidebar: API Keys (수정된 부분) ----------------
    with st.sidebar:
        st.header("🔑 API 설정")

        # 현재 저장된 상태 표시
        is_valid, validation_msg = validate_api_keys()
        if is_valid:
            st.success("✅ API 키가 저장되어 있습니다")
        else:
            st.warning("⚠️ API 키를 입력해주세요")

        method = st.radio(
            "입력 방식",
            ["개별 입력", "일괄 붙여넣기"],
            horizontal=True,
            key="api_input_method"
        )

        if method == "개별 입력":
            # 저장된 값을 입력 위젯에 반영
            sync_saved_to_inputs(force=False)

            st.subheader("네이버 데이터랩 API")
            datalab_id = st.text_input(
                "Client ID", 
                type="password", 
                value=st.session_state.get("datalab_id_input", ""),
                key="datalab_id_input"
            )
            datalab_secret = st.text_input(
                "Client Secret", 
                type="password", 
                value=st.session_state.get("datalab_secret_input", ""),
                key="datalab_secret_input"
            )

            st.subheader("네이버 광고 API")
            ads_key = st.text_input(
                "API Key", 
                type="password", 
                value=st.session_state.get("ads_key_input", ""),
                key="ads_key_input"
            )
            ads_secret = st.text_input(
                "Secret", 
                type="password", 
                value=st.session_state.get("ads_secret_input", ""),
                key="ads_secret_input"
            )
            ads_cid = st.text_input(
                "Customer ID", 
                type="password", 
                value=st.session_state.get("ads_cid_input", ""),
                key="ads_cid_input"
            )

        else:  # 일괄 붙여넣기
            st.subheader("일괄 붙여넣기")
            st.markdown("""
            **입력 형식 (둘 중 하나 선택):**
            
            **방법 1: 순서대로 5줄**
            ```
            your_datalab_client_id
            your_datalab_client_secret
            your_ads_api_key
            your_ads_secret
            your_ads_customer_id
            ```
            
            **방법 2: key=value 형식**
            ```
            datalab_client_id=your_id
            datalab_client_secret=your_secret
            ads_api_key=your_key
            ads_secret=your_secret
            ads_customer_id=your_customer_id
            ```
            """)
            
            bulk_text = st.text_area(
                "API 키 일괄 입력",
                height=150,
                value=st.session_state.get("bulk_text", ""),
                key="bulk_text",
                placeholder="위의 형식 중 하나로 API 키를 입력하세요..."
            )

            # 미리보기 기능
            if bulk_text.strip():
                with st.expander("🔍 입력 내용 미리보기"):
                    try:
                        parsed = parse_bulk_api_keys(bulk_text)
                        if parsed:
                            st.write("**인식된 API 키:**")
                            for key, value in parsed.items():
                                masked_value = value[:4] + "*" * (len(value) - 8) + value[-4:] if len(value) > 8 else "*" * len(value)
                                st.write(f"- {key}: {masked_value}")
                        else:
                            st.warning("인식된 API 키가 없습니다. 형식을 확인해주세요.")
                    except Exception as e:
                        st.error(f"파싱 오류: {e}")

        # API 키 저장 버튼
        if st.button("💾 API 키 저장", type="primary", use_container_width=True):
            try:
                if method == "일괄 붙여넣기":
                    bulk_text = st.session_state.get("bulk_text", "").strip()
                    if not bulk_text:
                        st.error("❌ 일괄 입력 내용이 비어있습니다.")
                        st.stop()
                    
                    parsed = parse_bulk_api_keys(bulk_text)
                    
                    # 디버그 출력
                    st.write("**디버그: 파싱 결과**")
                    st.write(parsed)
                    
                    if not parsed:
                        st.error("❌ API 키를 인식할 수 없습니다. 입력 형식을 확인해주세요.")
                        st.stop()
                    
                    # 각 키별 확인
                    required_keys = ["datalab_client_id", "datalab_client_secret", "ads_api_key", "ads_secret", "ads_customer_id"]
                    missing_keys = [key for key in required_keys if key not in parsed or not parsed[key]]
                    
                    if missing_keys:
                        st.error(f"❌ 누락된 키: {missing_keys}")
                        st.write("**파싱된 키들:**", list(parsed.keys()))
                        st.stop()

                    # 저장된 키에 업데이트
                    for key in ["datalab_client_id", "datalab_client_secret", "ads_api_key", "ads_secret", "ads_customer_id"]:
                        if key in parsed:
                            st.session_state[key] = parsed[key]
                        
                    # 개별 입력 위젯에도 반영
                    sync_saved_to_inputs(force=True)
                    
                    # 라디오 모드 변경 제거 - 사용자가 직접 전환하도록 함

                else:  # 개별 입력
                    # 개별 입력 값들을 저장 키에 저장
                    st.session_state["datalab_client_id"] = st.session_state.get("datalab_id_input", "").strip()
                    st.session_state["datalab_client_secret"] = st.session_state.get("datalab_secret_input", "").strip()
                    st.session_state["ads_api_key"] = st.session_state.get("ads_key_input", "").strip()
                    st.session_state["ads_secret"] = st.session_state.get("ads_secret_input", "").strip()
                    st.session_state["ads_customer_id"] = st.session_state.get("ads_cid_input", "").strip()

                # 유효성 검사
                is_valid, msg = validate_api_keys()
                if is_valid:
                    st.session_state.api_keys_saved = True
                    st.success("✅ API 키 저장 완료!")
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

            except Exception as e:
                st.error(f"❌ API 키 저장 중 오류 발생: {e}")

        # 저장된 키 확인 버튼 (디버깅용)
        if st.button("🔍 저장된 키 확인 (마스킹)"):
            st.write("**현재 저장된 API 키:**")
            for key in ["datalab_client_id", "datalab_client_secret", "ads_api_key", "ads_secret", "ads_customer_id"]:
                value = st.session_state.get(key, "")
                if value:
                    masked = value[:4] + "*" * (len(value) - 8) + value[-4:] if len(value) > 8 else "*" * len(value)
                    st.write(f"- {key}: {masked}")
                else:
                    st.write(f"- {key}: (비어있음)")






        st.markdown("---")
        st.markdown("""
        ### 📖 API 키 발급 안내
        **데이터랩 API**: 네이버 개발자센터 → 애플리케이션 등록 → 데이터랩(검색어트렌드)  
        **광고 API**: 네이버 광고관리자 → 도구 → API 관리 → API 키 발급
        """)

    # ---------------- Tabs ----------------
    tab1, tab2, tab3, tab4 = st.tabs(["📝 키워드 입력", "📊 분석 결과", "📈 추천 순위", "💡 가이드"])

    # ---- Tab1: 입력
    with tab1:
        st.header("분석할 키워드 입력")
        if not st.session_state.api_keys_saved:
            st.warning("⚠️ 먼저 사이드바에서 API 키를 입력/저장하세요.")

        col1, col2 = st.columns([2, 1])
        with col1:
            keywords_text = st.text_area(
                "키워드를 입력 (한 줄에 하나, 최대 30개)",
                height=280,
                placeholder="멜라토닌\n테아닌\n루테인\n밀크씨슬\n오메가3\n프로바이오틱스\n비타민D",
                help="띄어쓰기는 자동 제거됩니다. 예: '밀크 씨슬' → '밀크씨슬'"
            )
        with col2:
            st.subheader("📅 분석 기간")
            months_map = {12:"1년", 24:"2년", 36:"3년", 48:"4년", 60:"5년",
                          72:"6년", 84:"7년", 96:"8년", 108:"9년", 120:"10년"}
            months = st.select_slider(
                "기간 선택",
                options=list(months_map.keys()),
                value=12,
                format_func=lambda x: months_map[x]
            )
            st.info(f"📊 선택된 기간: **{months_map[months]} ({months}개월)**")

            st.markdown("---")
            if st.button("🔍 분석 시작", type="primary", use_container_width=True):
                datalab_client_id     = st.session_state.get("datalab_client_id","")
                datalab_client_secret = st.session_state.get("datalab_client_secret","")
                ads_api_key           = st.session_state.get("ads_api_key","")
                ads_secret            = st.session_state.get("ads_secret","")
                ads_customer_id       = st.session_state.get("ads_customer_id","")


                if not all([datalab_client_id, datalab_client_secret, ads_api_key, ads_secret, ads_customer_id]):
                    st.error("⚠️ 모든 API 키를 입력해주세요.")
                    st.stop()

                # 키워드 전처리: 공백 제거 + 중복 제거
                raw_lines = keywords_text.split("\n")
                keywords, dropped = unique_cleaned_list(raw_lines)
                if not keywords:
                    st.error("⚠️ 키워드를 입력해주세요.")
                    st.stop()
                if len(keywords) > 30:
                    st.warning(f"⚠️ 30개를 초과한 키워드는 제외됩니다. ({len(keywords)}개 → 30개)")
                    keywords = keywords[:30]
                if dropped:
                    st.info(f"ℹ️ 공백 제거 후 중복된 {len(dropped)}개 키워드는 자동 제외됨: {', '.join(sorted(set(dropped)))}")

                st.success(f"✅ {len(keywords)}개 키워드 분석을 시작합니다.")

                try:
                    with st.spinner("분석 중…"):
                        # 1) 절대 검색량
                        st.info("📍 1단계: 절대 검색량 조회")
                        absolute_volumes = get_absolute_volumes(keywords, ads_api_key, ads_secret, ads_customer_id)
                        valid_keywords = [kw for kw in keywords if absolute_volumes.get(kw, 0) > 0]
                        if not valid_keywords:
                            st.error("❌ 유효한 검색량을 가진 키워드가 없습니다.")
                            st.stop()

                        # 2) 추이(상대지수)
                        st.info("📍 2단계: 검색량 추이 조회")
                        trend_df = get_trend_data(valid_keywords, months, datalab_client_id, datalab_client_secret)

                        # 3) 절대값 변환
                        st.info("📍 3단계: 데이터 변환")
                        final_df = convert_to_absolute_values(trend_df, absolute_volumes)

                        # 4) 추천 점수
                        st.info("📍 4단계: 추천 점수 계산")
                        recommendations = calculate_recommendation_score(final_df, absolute_volumes, months)

                        st.session_state['final_df'] = final_df
                        st.session_state['absolute_volumes'] = absolute_volumes
                        st.session_state['recommendations'] = recommendations
                        st.session_state['months'] = months

                        st.success("🎉 분석 완료! '분석 결과' 또는 '추천 순위' 탭에서 확인하세요.")
                except Exception as e:
                    st.error(f"❌ 오류 발생: {e}")
                    import traceback
                    st.text(traceback.format_exc())

    # ---- Tab2: 분석 결과
    with tab2:
        st.header("📊 분석 결과")
        if 'final_df' in st.session_state:
            # 보기 옵션: 툴팁 모드 + 포커스 키워드
            _col1, _col2 = st.columns([1,1])
            with _col1:
                tooltip_mode = st.radio("툴팁 모드", ["개별", "통합"], horizontal=True,
                                        help="개별: 호버한 선만 / 통합: 같은 월의 모든 선을 한 패널에 표시")
                hover_mode = 'closest' if tooltip_mode == "개별" else 'x unified'
            with _col2:
                uniq_kws = list(st.session_state['final_df']['keyword'].unique())
                focus_kw = st.selectbox("포커스 키워드(선택)", ["(없음)"] + uniq_kws, index=0,
                                        help="선택 시 해당 선만 강조, 나머지는 희미하게 표시")

            fig = create_interactive_chart(st.session_state['final_df'], hover_mode=hover_mode)
            fig = apply_focus(fig, focus_kw)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # 상세 데이터
            st.subheader("📋 상세 데이터")
            selected_keywords = st.multiselect(
                "키워드 선택", st.session_state['final_df']['keyword'].unique(),
                default=st.session_state['final_df']['keyword'].unique()[:5]
            )
            if selected_keywords:
                filtered_df = st.session_state['final_df'][st.session_state['final_df']['keyword'].isin(selected_keywords)]
                pivot_df = filtered_df.pivot(index='period', columns='keyword', values='absolute_volume')
                st.dataframe(pivot_df.style.format("{:,.0f}"), use_container_width=True)

                # Excel 다운로드
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    (st.session_state.get('recommendations', pd.DataFrame())).to_excel(writer, sheet_name='추천순위', index=False)
                    st.session_state['final_df'].to_excel(writer, sheet_name='추이데이터', index=False)
                    summary = [{"키워드": k, "최근1개월_절대검색량": v} for k, v in st.session_state['absolute_volumes'].items()]
                    pd.DataFrame(summary).to_excel(writer, sheet_name='절대검색량', index=False)
                buffer.seek(0)
                st.download_button(
                    "📥 Excel 다운로드",
                    data=buffer,
                    file_name=f"keyword_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("👈 먼저 '키워드 입력' 탭에서 분석을 실행하세요.")

    # ---- Tab3: 추천 순위
    with tab3:
        st.header("🏆 제품 런칭 추천 원료 순위")

        # 세션에 없으면 즉석 계산(안전장치)
        if 'recommendations' not in st.session_state and 'final_df' in st.session_state and 'absolute_volumes' in st.session_state:
            try:
                st.session_state['recommendations'] = calculate_recommendation_score(
                    st.session_state['final_df'], st.session_state['absolute_volumes'], st.session_state.get('months', 12)
                )
            except Exception as e:
                st.error(f"추천 순위 계산 중 오류: {e}")

        if 'recommendations' in st.session_state and not st.session_state['recommendations'].empty:
            rec_df = st.session_state['recommendations']

            st.subheader("🎯 TOP 5 추천 원료")
            for _, row in rec_df.head(5).iterrows():
                grade_class = get_grade_class(row['등급'])
                c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 2, 2])
                with c1: st.metric("순위", f"#{int(row['순위'])}")
                with c2: st.metric("원료명", row['키워드'])
                with c3: st.metric("현재 검색량", f"{int(row['현재검색량']):,}회")
                with c4:
                    delta_color = "normal" if row['성장률(%)'] >= 0 else "inverse"
                    st.metric("성장률", f"{row['성장률(%)']:+.1f}%", delta_color=delta_color)
                with c5:
                    st.markdown(f"""
                    <div class='grade-card {grade_class}'>
                        <div style='font-size: 16px; margin-bottom: 5px;'>{row['등급']}</div>
                        <div style='font-size: 24px;'>{int(row['총점'])}점</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("---")

            st.subheader("📊 전체 순위")
            display_cols = ['순위', '키워드', '현재검색량', '성장률(%)', '총점', '등급']
            st.dataframe(rec_df[display_cols].style.format({
                '현재검색량': '{:,.0f}', '성장률(%)': '{:+.1f}', '총점': '{:.0f}'
            }), use_container_width=True, height=420)

            with st.expander("📋 점수 상세 보기"):
                detail_cols = ['키워드', '검색량점수', '성장률점수', '안정성점수', '트렌드점수', '변동계수', '최근6개월평균']
                st.dataframe(rec_df[detail_cols], use_container_width=True)
        else:
            st.info("👈 먼저 '키워드 입력' 탭에서 분석을 실행하세요. (필요시 탭 진입 시 자동 재계산)")

    # ---- Tab4: 초보자 가이드 (원문 유지)
    with tab4:
        st.header("💡 초보자를 위한 상세 가이드")
     
        st.markdown("""
        ### 🎯 이 프로그램은 무엇을 하나요?
        
        이 프로그램은 **네이버에서 특정 원료(키워드)가 얼마나 검색되는지**를 분석해서,
        **어떤 원료로 제품을 만들면 좋을지** 추천해주는 도구입니다.
        
        예를 들어, "루테인"이라는 원료가 최근 검색량이 늘고 있다면,
        루테인 제품을 출시하면 성공 가능성이 높다는 의미입니다!
        
        ---
        
        ### 📊 점수는 어떻게 계산되나요?
        
        총 100점 만점으로, 4가지 항목을 평가합니다:
        
        #### 1️⃣ **검색량 점수 (40점)** - "얼마나 많이 검색되나요?"
        
        한 달에 사람들이 이 키워드를 검색한 횟수를 봅니다.
        
        - **40점 (최고!)**: 월 10,000회 이상
          - 예: "비타민D"처럼 매우 인기 있는 원료
          - 이미 많은 사람이 알고 찾는 대중적인 원료
        
        - **30점 (좋음)**: 5,000~10,000회
          - 예: "밀크씨슬"처럼 꽤 알려진 원료
          - 충분한 수요가 있어 제품화하기 좋음
        
        - **20점 (보통)**: 1,000~5,000회
          - 예: "보스웰리아"처럼 어느 정도 알려진 원료
          - 타겟 마케팅이 필요하지만 가능성 있음
        
        - **10점 (낮음)**: 1,000회 미만
          - 아직 대중에게 잘 알려지지 않은 원료
          - 니치 마켓을 노릴 수 있지만 신중한 접근 필요
        
        #### 2️⃣ **성장률 점수 (25점)** - "검색량이 늘고 있나요?"
        
        처음과 끝을 비교해서 얼마나 성장했는지 봅니다.
        
        - **25점 (폭발적!)**: 50% 이상 성장
          - 예: 6개월 전 1,000회 → 현재 1,500회 이상
          - 핫한 트렌드! 빠르게 대응하면 좋은 기회
        
        - **20점 (성장중)**: 20~50% 성장
          - 꾸준히 인기가 올라가는 중
          - 안정적으로 성장하는 시장
        
        - **15점 (유지)**: 0~20% 성장
          - 큰 변화는 없지만 안정적
          - 검증된 시장, 경쟁은 있을 수 있음
        
        - **10점 (주의)**: -20~0% 하락
          - 약간 관심이 줄어드는 중
          - 시장 상황을 더 조사해봐야 함
        
        - **5점 (위험)**: -20% 이상 하락
          - 인기가 많이 떨어지는 중
          - 제품화는 신중하게 결정
        
        #### 3️⃣ **안정성 점수 (20점)** - "검색량이 들쭉날쭉한가요?"
        
        변동계수(CV)로 측정합니다. 쉽게 말해 "얼마나 일정한가"를 봅니다.
        
        - **20점 (매우 안정)**: 변동계수 0.3 이하
          - 매달 비슷하게 검색됨
          - 예측 가능한 안정적 수요
          - 재고 관리가 쉬움
        
        - **15점 (안정)**: 변동계수 0.3~0.5
          - 약간의 변동은 있지만 괜찮음
          - 계절 영향을 조금 받을 수 있음
        
        - **10점 (보통)**: 변동계수 0.5~0.8
          - 변동이 있는 편
          - 이벤트나 계절에 영향받음
        
        - **5점 (불안정)**: 변동계수 0.8 초과
          - 매우 불규칙한 패턴
          - 일시적 유행일 가능성
          - 신중한 접근 필요
        
        #### 4️⃣ **트렌드 점수 (15점)** - "최근에 더 인기가 있나요?"
        
        최근 6개월과 그 이전 6개월을 비교합니다.
        
        - **15점 (급상승!)**: 20% 이상 상승
          - 최근 들어 급격히 인기 상승
          - 지금이 진입 적기일 수 있음
        
        - **12점 (상승중)**: 10~20% 상승
          - 최근 관심이 늘어나는 중
          - 좋은 타이밍
        
        - **10점 (유지)**: 0~10% 상승
          - 꾸준한 관심 유지
          - 안정적인 시장
        
        - **7점 (주의)**: -10~0% 하락
          - 최근 관심이 조금 줄어듦
          - 원인 파악 필요
        
        - **5점 (하락)**: -10% 이상 하락
          - 최근 인기가 많이 떨어짐
          - 다른 원료를 고려해보세요
        
        ---
        
        ### 🏆 등급의 의미와 추천 전략
        
        #### **A+ (85점 이상)** 🌟 강력추천 - "지금 당장 시작하세요!"
        - **의미**: 검색량도 많고, 성장하고 있으며, 안정적인 최고의 원료
        - **전략**: 
          - 즉시 제품 개발 착수
          - 메인 제품 라인으로 개발
          - 적극적인 마케팅 투자 권장
        - **예시**: 현재 핫한 트렌드 원료들
        
        #### **A (75~84점)** ✅ 추천 - "좋은 기회입니다"
        - **의미**: 전반적으로 우수한 지표를 보이는 원료
        - **전략**:
          - 제품 개발 적극 검토
          - 시장 조사 후 빠른 출시
          - 온라인 마케팅 집중
        
        #### **B+ (65~74점)** 🤔 고려 - "가능성은 있어요"
        - **의미**: 일부 지표는 좋지만 약점도 있는 원료
        - **전략**:
          - 추가 시장 조사 필요
          - 차별화 전략 수립 중요
          - 테스트 제품으로 시작
        
        #### **B (55~64점)** ⚠️ 보통 - "신중하게 접근하세요"
        - **의미**: 평균적인 수준, 리스크와 기회 공존
        - **전략**:
          - 니치 마켓 공략
          - 번들 상품으로 구성
          - 소량 생산으로 테스트
        
        #### **C (55점 미만)** ❌ 신중 - "다시 생각해보세요"
        - **의미**: 현재는 추천하지 않는 원료
        - **전략**:
          - 다른 원료 검토 권장
          - 시장 상황 변화 모니터링
          - 혁신적 차별화 없이는 진입 보류
        
        ---
        
        ### 📈 실전 활용 팁
        
        #### 1. **키워드 선정 꿀팁** 🍯
        
        **좋은 키워드 예시:**
        - ✅ "루테인" (정확한 원료명)
        - ✅ "밀크씨슬" (띄어쓰기 없이)
        - ✅ "오메가3" (숫자 붙여서)
        
        **괜찮은 키워드 (자동 처리됨):**
        - ✅ "밀크 씨슬" → 자동으로 "밀크씨슬"로 변환
        - ✅ "비타민 D" → 자동으로 "비타민D"로 변환
        
        **피해야 할 키워드:**
        - ❌ "눈 영양제" (너무 포괄적)
        - ❌ "루테인 효능" (원료명만 입력)
        - ❌ "ㄹㅌㅇ" (축약어 사용 금지)
        
        #### 2. **분석 기간 선택 가이드** 📅
        
        - **1년 분석**: 최신 트렌드, 신규 원료
          - 빠르게 변하는 시장에 적합
          - 예: 최근 유행하는 신소재
        
        - **2년 분석**: 일반적인 분석
          - 대부분의 원료에 적합
          - 계절성 파악 가능
        
        - **3~4년 분석**: 중장기 트렌드
          - 안정성 중시할 때
          - 대규모 투자 전 확인용
        
        - **5년 분석**: 장기 시장 분석
          - 시장의 큰 흐름 파악
          - 브랜드 전략 수립용
        
        #### 3. **결과 해석 주의사항** ⚠️
        
        1. **검색량이 낮아도 포기하지 마세요**
           - 성장률이 높으면 미래 가능성 있음
           - 니치 마켓도 수익성 좋을 수 있음
        
        2. **A+ 등급이어도 추가 확인 필요**
           - 경쟁사 제품 조사
           - 원료 가격 동향 확인
           - 법적 규제 사항 체크
        
        3. **변동계수가 높은 경우**
           - 계절 상품일 가능성 (예: 비타민D - 겨울)
           - 이슈에 민감한 원료 (방송 영향 등)
           - 재고 관리 전략 필요
        
        ---
        
        ### 🔍 추가로 확인해야 할 사항들
        
        #### **시장 조사** 📊
        1. 네이버 쇼핑에서 검색 결과 수 확인
        2. 경쟁 제품 가격대 조사
        3. 리뷰 분석으로 소비자 니즈 파악
        
        #### **원료 조사** 🧪
        1. 식약처 인정 원료 여부
        2. 원료 수급 안정성
        3. 원가 및 마진 구조
        
        #### **타겟 분석** 👥
        1. 주 구매 연령층 확인
        2. 성별 선호도 차이
        3. 구매 채널 선호도
        
        ---
        
        ### 🚀 성공 사례
        
        **사례 1: 루테인** 
        - 2018년부터 검색량 급증
        - 스마트폰 사용 증가와 맞물려 성장
        - 조기 진입한 브랜드들이 시장 선점
        
        **사례 2: 밀크씨슬**
        - 음주 문화와 연결되어 꾸준한 수요
        - 안정적인 검색량으로 steady seller
        
        ---
        
        ### ❓ 자주 묻는 질문
        
        **Q1. 검색량이 0인 키워드는 왜 그런가요?**
        - 너무 생소한 원료이거나
        - 오타가 있거나
        - 다른 이름으로 더 많이 검색됨
        
        **Q2. 같은 원료인데 이름이 여러 개면?**
        - 각각 따로 입력해서 비교
        - 예: "비타민C", "아스코르브산"
        
        **Q3. 얼마나 자주 분석해야 하나요?**
        - 월 1회 정도 추천
        - 신제품 출시 전 필수 체크
        
        **Q4. API 키는 한 번만 입력하면 되나요?**
        - 네! 사이드바에서 API 키를 입력하고 "API 키 저장" 버튼을 누르면
        - 프로그램을 다시 실행해도 저장된 값이 유지됩니다
        
        ---
        
        ### 📞 도움이 필요하신가요?
        
        이 가이드를 읽고도 궁금한 점이 있다면,
        강의 커뮤니티에 질문을 남겨주세요!
        
        모든 질문은 초보자의 관점에서 친절하게 답변드립니다. 😊
        
        **함께 성공적인 제품을 만들어봐요! 🎉**
        """)

if __name__ == "__main__":
    main()
