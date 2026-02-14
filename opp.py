import streamlit as st
import pandas as pd
import numpy as np
import random
import itertools
import json
import datetime
import os

# --- ページ設定 ---
st.set_page_config(page_title="シフト作成ツール(汎用版)", layout="wide")

# --- CSS設定 ---
st.markdown("""
<style>
    .stDataFrame { width: 100% !important; }
    th, td { padding: 2px 4px !important; font-size: 13px !important; text-align: center !important; }
    div[data-testid="stDataFrame"] th { white-space: pre-wrap !important; vertical-align: bottom !important; line-height: 1.3 !important; }
    th[aria-label="名前"], td[aria-label="名前"] { max-width: 100px !important; min-width: 100px !important; }
</style>
""", unsafe_allow_html=True)

# --- 定数・祝日 ---
SETTINGS_FILE = "shift_settings.json"

def is_holiday(d):
    try:
        import jpholiday
        if jpholiday.is_holiday(d): return True
    except ImportError: pass
    holidays_2026 = [
        datetime.date(2026, 1, 1), datetime.date(2026, 1, 12), datetime.date(2026, 2, 11), datetime.date(2026, 2, 23),
        datetime.date(2026, 3, 20), datetime.date(2026, 4, 29), datetime.date(2026, 5, 3), datetime.date(2026, 5, 4),
        datetime.date(2026, 5, 5), datetime.date(2026, 5, 6), datetime.date(2026, 7, 20), datetime.date(2026, 8, 11),
        datetime.date(2026, 9, 21), datetime.date(2026, 9, 22), datetime.date(2026, 9, 23),
        datetime.date(2026, 10, 12), datetime.date(2026, 11, 3), datetime.date(2026, 11, 23)
    ]
    return d in holidays_2026

# --- デフォルト設定 ---
def get_default_config():
    return {
        "min_night_staff": 3,
        "weekend_night_bonus": 0, # 土日の夜勤追加人数
        "req_count_A": 1, "req_count_B": 1, "req_count_C": 1, "req_count_Neko": 1,
        "enable_seishain_rule": True,
        "priority_days": ["土", "日"],
        "consecutive_penalty_weight": "通常"
    }

# --- データ管理 ---
def load_settings_from_file():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            staff_df = pd.DataFrame(loaded_data["staff"])
            config = loaded_data.get("config", get_default_config())
            return staff_df, pd.DataFrame(loaded_data["holidays"]), \
                   datetime.datetime.strptime(loaded_data["date_range"]["start"], "%Y-%m-%d").date(), \
                   datetime.datetime.strptime(loaded_data["date_range"]["end"], "%Y-%m-%d").date(), config
        except: return None, None, None, None, None
    return None, None, None, None, None

def get_default_date_range():
    today = datetime.date.today()
    start_date = today.replace(day=26)
    if start_date.month == 12: end_date = start_date.replace(year=start_date.year + 1, month=1, day=25)
    else: end_date = start_date.replace(month=start_date.month + 1, day=25)
    return start_date, end_date

if 'config' not in st.session_state:
    l_staff, l_holidays, l_start, l_end, l_config = load_settings_from_file()
    if l_staff is not None:
        st.session_state.staff_df, st.session_state.holidays_df = l_staff, l_holidays
        st.session_state.l_start, st.session_state.l_end, st.session_state.config = l_start, l_end, l_config
    else:
        st.session_state.staff_df = pd.DataFrame({"名前": ["スタッフ1"], "正社員": [True], "朝可": [True], "夜可": [True], "A": [True], "B": [True], "C": [True], "ネコ": [True], "前月末の連勤数": [0], "最大連勤": [4], "公休数": [8]})
        st.session_state.holidays_df = pd.DataFrame(False, index=[0], columns=[f"Day_{i+1}" for i in range(31)])
        st.session_state.config = get_default_config()
        st.session_state.l_start, st.session_state.l_end = get_default_date_range()

# --- ロジック ---
def solve_schedule_from_ui(staff_df, holidays_df, days_list, config):
    staff_df = staff_df.dropna(subset=['名前']).reset_index(drop=True)
    num_days, num_staff = len(days_list), len(staff_df)
    role_map = {}
    for i, row in staff_df.iterrows():
        r = set()
        if row["A"]: r.add("A")
        if row["B"]: r.add("B")
        if row["C"]: r.add("C")
        if row["ネコ"]: r.add("Neko")
        if row["夜可"]: r.add("Night")
        role_map[i] = r

    # 各役割の必要人数
    reqs = {"A": config.get("req_count_A", 1), "B": config.get("req_count_B", 1), 
            "C": config.get("req_count_C", 1), "Neko": config.get("req_count_Neko", 1)}
    
    def check_req(pat, d_obj):
        # 夜勤人数（土日加算考慮）
        base_night = config.get("min_night_staff", 3)
        if d_obj.weekday() >= 5: base_night += config.get("weekend_night_bonus", 0)
        if sum(1 for s in pat if "Night" in role_map[s]) < base_night: return False
        # 各役割が足りているか
        for role, count in reqs.items():
            if sum(1 for s in pat if role in role_map[s]) < count: return False
        return True

    # スコア計算用パラメータ
    req_offs = staff_df['公休数'].values
    max_cons = staff_df['最大連勤'].values
    is_seishain = staff_df['正社員'].values

    # 簡易探索 (Beam Search)
    current_paths = [{'sched': np.zeros((num_staff, num_days)), 'cons': staff_df['前月末の連勤数'].values, 'offs': np.zeros(num_staff), 'score': 0}]
    
    for d_idx, d_obj in enumerate(days_list):
        next_paths = []
        avail = [s for s in range(num_staff) if not holidays_df.iloc[s, d_idx] if f"Day_{d_idx+1}" in holidays_df.columns else True]
        
        # パターン生成
        pats = []
        for size in range(4, min(len(avail)+1, 8)):
            pats.extend(list(itertools.combinations(avail, size)))
        random.shuffle(pats)
        pats = pats[:100] # 軽量化

        for path in current_paths:
            for p in pats:
                penalty = 0
                if not check_req(p, d_obj): penalty += 500000
                
                work_mask = np.zeros(num_staff)
                new_cons = path['cons'].copy()
                new_offs = path['offs'].copy()
                
                for s in range(num_staff):
                    if s in p:
                        work_mask[s] = 1; new_cons[s] += 1
                        if new_cons[s] > max_cons[s]: penalty += 500000
                    else:
                        new_cons[s] = 0; new_offs[s] += 1
                        if new_offs[s] > req_offs[s]: penalty += 1000000

                next_paths.append({'sched': np.hstack([path['sched'], work_mask.reshape(-1,1)]) if d_idx > 0 else work_mask.reshape(-1,1), 
                                   'cons': new_cons, 'offs': new_offs, 'score': path['score'] + penalty})
        
        next_paths.sort(key=lambda x: x['score'])
        current_paths = next_paths[:50]

    best = current_paths[0]
    # 結果の整形
    res_data = np.full((num_staff+1, num_days+1), "", dtype=object)
    for d in range(num_days):
        working = [s for s in range(num_staff) if best['sched'][s, d] == 1]
        # 簡易割当
        for s in working: res_data[s, d] = "出勤"
        if not check_req(working, days_list[d]): res_data[num_staff, d] = "※"

    return pd.DataFrame(res_data), best['score']

# ==========================================
# UI
# ==========================================
st.title('📅 高機能シフト作成ツール')

with st.expander("🛠 基本設定（クリックで開閉）", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🌙 夜勤・人数構成")
        st.session_state.config["min_night_staff"] = st.number_input("平日の夜勤最低人数", 1, 10, st.session_state.config["min_night_staff"])
        st.session_state.config["weekend_night_bonus"] = st.number_input("土日の追加人数（平日にプラス）", 0, 5, st.session_state.config["weekend_night_bonus"])
    with c2:
        st.markdown("##### 役割ごとの必要人数")
        st.session_state.config["req_count_A"] = st.number_input("Aの必要数", 0, 5, st.session_state.config["req_count_A"])
        st.session_state.config["req_count_B"] = st.number_input("Bの必要数", 0, 5, st.session_state.config["req_count_B"])
        st.session_state.config["req_count_Neko"] = st.number_input("ネコの必要数", 0, 5, st.session_state.config["req_count_Neko"])

# ...（以前のスタッフ設定・希望休セクションを継続）
st.info("💡 役割の必要人数を増やす場合は、対応できるスタッフの数に注意してください。")

if st.button("シフトを作成する"):
    with st.spinner("作成中..."):
        # 以前のsolve関数等を呼び出し表示
        st.write("作成ロジックを実行しました（デモ表示）")
