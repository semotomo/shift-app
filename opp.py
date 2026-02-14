import streamlit as st
import pandas as pd
import numpy as np
import random
import itertools
import json
import datetime
import os

# --- ページ設定 ---
st.set_page_config(page_title="シフト作成ツール", layout="wide")

# --- CSS設定 ---
st.markdown("""
<style>
    .stDataFrame { width: 100% !important; }
    th, td { padding: 2px 4px !important; font-size: 13px !important; text-align: center !important; }
    div[data-testid="stDataFrame"] th { white-space: pre-wrap !important; vertical-align: bottom !important; line-height: 1.3 !important; }
    th[aria-label="名前"], td[aria-label="名前"] { max-width: 100px !important; min-width: 100px !important; }
</style>
""", unsafe_allow_html=True)

# --- 定数 ---
SETTINGS_FILE = "shift_settings.json"

# --- 祝日判定関数 ---
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

# --- データ読み込み ---
def load_settings_from_file():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            staff_df = pd.DataFrame(loaded_data["staff"])
            cols_def = {"正社員": False, "朝可": True, "夜可": False, "A": False, "B": False, "C": False, "ネコ": False, "最大連勤": 4, "レベル": "スタッフ"}
            for col, val in cols_def.items():
                if col not in staff_df.columns: staff_df[col] = val
            start_d, end_d = None, None
            if "date_range" in loaded_data:
                try:
                    start_d = datetime.datetime.strptime(loaded_data["date_range"]["start"], "%Y-%m-%d").date()
                    end_d = datetime.datetime.strptime(loaded_data["date_range"]["end"], "%Y-%m-%d").date()
                except: pass
            config = loaded_data.get("config", {"min_night_staff": 3, "enable_seishain_rule": True, "priority_days": ["土", "日"], "consecutive_penalty_weight": "通常"})
            pairs_df = pd.DataFrame(loaded_data.get("pairs", []))
            if pairs_df.empty: pairs_df = pd.DataFrame(columns=["Staff A", "Staff B", "Type"])
            return staff_df, pd.DataFrame(loaded_data["holidays"]), start_d, end_d, config, pairs_df
        except: return None, None, None, None, None, None
    return None, None, None, None, None, None

def get_default_date_range():
    today = datetime.date.today()
    start_date = today.replace(day=26)
    if start_date.month == 12: end_date = start_date.replace(year=start_date.year + 1, month=1, day=25)
    else: end_date = start_date.replace(month=start_date.month + 1, day=25)
    return start_date, end_date

if 'staff_df' not in st.session_state:
    l_staff, l_holidays, l_start, l_end, l_config, l_pairs = load_settings_from_file()
    if l_staff is not None:
        st.session_state.staff_df, st.session_state.holidays_df, st.session_state.loaded_start_date, st.session_state.loaded_end_date, st.session_state.config, st.session_state.pairs_df = l_staff, l_holidays, l_start, l_end, l_config, l_pairs
    else:
        st.session_state.staff_df = pd.DataFrame({"名前": ["スタッフ1"], "レベル": ["スタッフ"], "正社員": [True], "朝可": [True], "夜可": [True], "A": [True], "B": [True], "C": [True], "ネコ": [True], "前月末の連勤数": [0], "最大連勤": [4], "公休数": [8]})
        st.session_state.holidays_df = pd.DataFrame(False, index=[0], columns=[f"Day_{i+1}" for i in range(31)])
        st.session_state.config = {"min_night_staff": 3, "enable_seishain_rule": True, "priority_days": ["土", "日"], "consecutive_penalty_weight": "通常"}
        st.session_state.pairs_df = pd.DataFrame(columns=["Staff A", "Staff B", "Type"])

# --- ロジック関数 ---
def can_cover_required_roles(staff_list, role_map, level_map, min_night_count=3):
    if sum(1 for s in staff_list if "Night" in role_map[s]) < min_night_count: return False
    if sum(1 for s in staff_list if level_map[s] == "リーダー") < 1: return False
    neko_cands = [s for s in staff_list if "Neko" in role_map[s]]
    p_neko = [s for s in neko_cands if "A" not in role_map[s] and "B" not in role_map[s]]
    neko_fixed = p_neko[0] if p_neko else (neko_cands[0] if neko_cands else None)
    if neko_fixed is not None:
        rem = [x for x in staff_list if x != neko_fixed]
        if len(rem) < 3: return False
        for p in itertools.permutations(rem, 3):
             if 'A' in role_map[p[0]] and 'B' in role_map[p[1]] and 'C' in role_map[p[2]]: return True
        return False
    else:
        if len(staff_list) < 4: return False
        for p in itertools.permutations(staff_list, 4):
             if 'Neko' in role_map[p[0]] and 'A' in role_map[p[1]] and 'B' in role_map[p[2]] and 'C' in role_map[p[3]]: return True
        return False

def assign_roles_smartly(working_indices, role_map):
    assignments = {}
    pool = list(working_indices)
    neko_cands = [s for s in pool if "Neko" in role_map[s]]
    p_neko = [s for s in neko_cands if "A" not in role_map[s] and "B" not in role_map[s]]
    neko_fixed = p_neko[0] if p_neko else (neko_cands[0] if neko_cands else None)
    
    found = False
    if neko_fixed is not None:
        rem = [x for x in pool if x != neko_fixed]
        for p in itertools.permutations(rem, 3):
            if 'A' in role_map[p[0]] and 'B' in role_map[p[1]] and 'C' in role_map[p[2]]:
                assignments[neko_fixed] = 'ネコ'; assignments[p[0]] = 'A'; assignments[p[1]] = 'B'; assignments[p[2]] = 'C'; found = True
                for ex in [x for x in rem if x not in p]: assignments[ex] = '〇'
                break
    if not found:
        for p in itertools.permutations(pool, 4):
            if 'Neko' in role_map[p[0]] and 'A' in role_map[p[1]] and 'B' in role_map[p[2]] and 'C' in role_map[p[3]]:
                assignments[p[0]] = 'ネコ'; assignments[p[1]] = 'A'; assignments[p[2]] = 'B'; assignments[p[3]] = 'C'; found = True
                for ex in [x for x in pool if x not in p]: assignments[ex] = '〇'
                break
    if not found:
        for s in pool: assignments[s] = '〇'
    return assignments

def solve_schedule_from_ui(staff_df, holidays_df, days_list, config, pairs_df):
    staff_df = staff_df.dropna(subset=['名前']).reset_index(drop=True)
    num_days, num_staff = len(days_list), len(staff_df)
    role_map = {}; level_map = {}
    for i, row in staff_df.iterrows():
        r = set()
        if row["A"]: r.add("A"); 
        if row["B"]: r.add("B"); 
        if row["C"]: r.add("C"); 
        if row["ネコ"]: r.add("Neko"); 
        if row["夜可"]: r.add("Night")
        role_map[i] = r; level_map[i] = row["レベル"]

    name_to_idx = {name: i for i, name in enumerate(staff_df['名前'])}
    pair_constraints = []
    if not pairs_df.empty:
        for _, row in pairs_df.iterrows():
            if row["Staff A"] in name_to_idx and row["Staff B"] in name_to_idx:
                pair_constraints.append({"a": name_to_idx[row["Staff A"]], "b": name_to_idx[row["Staff B"]], "type": row["Type"]})

    initial_cons = pd.to_numeric(staff_df['前月末の連勤数'], errors='coerce').fillna(0).astype(int).values
    req_offs = pd.to_numeric(staff_df['公休数'], errors='coerce').fillna(0).astype(int).values
    max_cons_limits = pd.to_numeric(staff_df['最大連勤'], errors='coerce').fillna(4).astype(int).values

    current_paths = [{'sched': np.zeros((num_staff, num_days), dtype=int), 'cons': initial_cons.copy(), 'offs': np.zeros(num_staff, dtype=int), 'score': 0}]
    BEAM_WIDTH = 200

    for d in range(num_days):
        next_paths = []
        avail = [s for s in range(num_staff) if not holidays_df.iloc[s, d] if f"Day_{d+1}" in holidays_df.columns else True]
        
        # 探索パターンの生成
        pats = []
        for size in range(min(len(avail), 4), min(len(avail)+1, 10)):
            pats.extend(list(itertools.combinations(avail, size)))
        random.shuffle(pats)
        use_patterns = pats[:150]

        for path in current_paths:
            for pat in use_patterns:
                penalty = 0
                new_offs = path['offs'].copy()
                new_cons = path['cons'].copy()
                
                # 1. 公休数厳守 (超重要)
                for s in range(num_staff):
                    if s not in pat: new_offs[s] += 1
                days_left = num_days - 1 - d
                for s in range(num_staff):
                    if new_offs[s] > req_offs[s]: penalty += 5000000 # 休みすぎ厳禁
                    if new_offs[s] + days_left < req_offs[s]: penalty += 5000000 # 出勤しすぎ厳禁
                
                # 2. 役割要件
                if not can_cover_required_roles(pat, role_map, level_map, config["min_night_staff"]):
                    penalty += 50000 # 足りない場合はスコアを下げる（が、公休優先のためエラーにはしない）

                # 3. 連勤
                for s in range(num_staff):
                    if s in pat:
                        new_cons[s] += 1
                        if new_cons[s] > max_cons_limits[s]: penalty += 100000
                    else: new_cons[s] = 0

                # 4. ペア制約
                for pc in pair_constraints:
                    a_in, b_in = (pc["a"] in pat), (pc["b"] in pat)
                    if pc["type"] == "NG" and a_in and b_in: penalty += 100000
                    if pc["type"] == "Pair" and a_in != b_in: penalty += 100000

                new_sched = path['sched'].copy(); new_sched[:, d] = [1 if s in pat else 0 for s in range(num_staff)]
                next_paths.append({'sched': new_sched, 'cons': new_cons, 'offs': new_offs, 'score': path['score'] + penalty})
        
        next_paths.sort(key=lambda x: x['score'])
        current_paths = next_paths[:BEAM_WIDTH]

    best = current_paths[0]
    res_data = np.full((num_staff + 1, num_days + 1), "", dtype=object)
    for d in range(num_days):
        working = [s for s in range(num_staff) if best['sched'][s, d] == 1]
        roles = assign_roles_smartly(working, role_map)
        for s, role in roles.items(): res_data[s, d] = role
        for s in range(num_staff):
            if best['sched'][s, d] == 0: res_data[s, d] = '／'
        if not can_cover_required_roles(working, role_map, level_map, config["min_night_staff"]): res_data[num_staff, d] = "※"

    for s in range(num_staff):
        res_data[s, num_days] = f"{num_days - best['offs'][s]}({best['offs'][s]})"
    return pd.DataFrame(res_data, columns=pd.MultiIndex.from_arrays([[str(d.day) for d in days_list]+["計"], ["祝" if is_holiday(d) else "日月火水木金土"[d.weekday()] for d in days_list]+[""]]), index=list(staff_df['名前'])+["不足"])

# --- UI実装 ---
st.title('📅 シフト作成ツール')

with st.sidebar:
    st.header("⚙️ 設定・読込")
    if st.button("💾 サーバーに保存"):
        save_dict = {"staff": st.session_state.staff_df.to_dict(), "holidays": st.session_state.holidays_df.to_dict(), "config": st.session_state.config, "pairs": st.session_state.pairs_df.to_dict(), "date_range": {"start": str(st.session_state.get("start_date", "")), "end": str(st.session_state.get("end_date", ""))}}
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(save_dict, f, ensure_ascii=False, indent=2)
        st.success("保存完了")
    
    start_date = st.date_input("開始日", get_default_date_range()[0])
    end_date = st.date_input("終了日", get_default_date_range()[1])
    days_list = pd.date_range(start_date, end_date).tolist()

with st.form("main_form"):
    with st.expander("🛠 基本・ペア設定"):
        st.session_state.config["min_night_staff"] = st.number_input("夜勤最低人数", 1, 10, st.session_state.config["min_night_staff"])
        st.session_state.pairs_df = st.data_editor(st.session_state.pairs_df, num_rows="dynamic", use_container_width=True, column_config={"Staff A": st.column_config.SelectboxColumn(options=st.session_state.staff_df["名前"].tolist()), "Staff B": st.column_config.SelectboxColumn(options=st.session_state.staff_df["名前"].tolist()), "Type": st.column_config.SelectboxColumn(options=["NG", "Pair"])})

    st.markdown("### 1️⃣ スタッフ設定")
    st.session_state.staff_df = st.data_editor(st.session_state.staff_df, num_rows="dynamic", use_container_width=True)
    
    st.markdown("### 2️⃣ 希望休入力")
    display_holidays = st.session_state.holidays_df.copy().reindex(index=range(len(st.session_state.staff_df)), columns=[f"Day_{i+1}" for i in range(len(days_list))], fill_value=False)
    edited_holidays = st.data_editor(display_holidays, use_container_width=True)
    
    if st.form_submit_button("✅ 反映して保存"):
        st.session_state.holidays_df = edited_holidays
        st.rerun()

if st.button("シフトを作成する"):
    with st.spinner("作成中..."):
        result_df = solve_schedule_from_ui(st.session_state.staff_df, st.session_state.holidays_df, days_list, st.session_state.config, st.session_state.pairs_df)
        st.dataframe(result_df.style.applymap(lambda v: 'background-color: #ffcccc' if v == '／' else ('background-color: #ff0000; color: white' if v == '※' else ''), axis=None), use_container_width=True)
