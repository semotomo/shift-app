import streamlit as st
import pandas as pd
import numpy as np
import random
import itertools
import json
import datetime
import os

# --- ページ設定 ---
st.set_page_config(page_title="シフト作成ツール(入力版)", layout="wide")

# --- CSS設定 ---
st.markdown("""
<style>
    .stDataFrame { width: 100% !important; }
    th, td {
        padding: 2px 4px !important;
        font-size: 13px !important;
        text-align: center !important; 
    }
    div[data-testid="stDataFrame"] th {
        white-space: pre-wrap !important;
        vertical-align: bottom !important;
        line-height: 1.3 !important;
    }
    div[data-testid="stDataFrame"] th span {
        white-space: pre-wrap !important;
        display: inline-block !important;
    }
    th[aria-label="名前"], td[aria-label="名前"] { max-width: 100px !important; min-width: 100px !important; }
    th[aria-label="社員"], td[aria-label="社員"],
    th[aria-label="朝"], td[aria-label="朝"],
    th[aria-label="夜"], td[aria-label="夜"],
    th[aria-label="A"], td[aria-label="A"],
    th[aria-label="B"], td[aria-label="B"],
    th[aria-label="C"], td[aria-label="C"],
    th[aria-label="🐱"], td[aria-label="🐱"] {
        max-width: 25px !important; min-width: 25px !important;
    }
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
        datetime.date(2026, 1, 1), datetime.date(2026, 1, 12),
        datetime.date(2026, 2, 11), datetime.date(2026, 2, 23),
        datetime.date(2026, 3, 20), datetime.date(2026, 4, 29),
        datetime.date(2026, 5, 3), datetime.date(2026, 5, 4), datetime.date(2026, 5, 5), datetime.date(2026, 5, 6),
        datetime.date(2026, 7, 20), datetime.date(2026, 8, 11),
        datetime.date(2026, 9, 21), datetime.date(2026, 9, 22), datetime.date(2026, 9, 23),
        datetime.date(2026, 10, 12), datetime.date(2026, 11, 3), datetime.date(2026, 11, 23)
    ]
    return d in holidays_2026

# --- デフォルト設定 ---
def get_default_config():
    return {
        "min_night_staff": 3, "enable_seishain_rule": True, "priority_days": ["土", "日"], "consecutive_penalty_weight": "通常" 
    }

# --- データ読み込み ---
def load_settings_from_file():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            staff_df = pd.DataFrame(loaded_data["staff"])
            for col in ["正社員", "朝可", "夜可", "A", "B", "C", "ネコ", "最大連勤"]:
                if col not in staff_df.columns:
                    staff_df[col] = (4 if col == "最大連勤" else (True if col == "朝可" else False))
            start_d, end_d = None, None
            if "date_range" in loaded_data:
                try:
                    start_d = datetime.datetime.strptime(loaded_data["date_range"]["start"], "%Y-%m-%d").date()
                    end_d = datetime.datetime.strptime(loaded_data["date_range"]["end"], "%Y-%m-%d").date()
                except: pass
            config = loaded_data.get("config", get_default_config())
            return staff_df, pd.DataFrame(loaded_data["holidays"]), start_d, end_d, config
        except Exception: return None, None, None, None, None
    return None, None, None, None, None

def get_default_date_range():
    today = datetime.date.today()
    start_date = today.replace(day=26)
    if start_date.month == 12: end_date = start_date.replace(year=start_date.year + 1, month=1, day=25)
    else: end_date = start_date.replace(month=start_date.month + 1, day=25)
    return start_date, end_date

# --- ロジック関数 ---
def get_role_map_from_df(staff_df):
    role_map = {}
    for i, row in staff_df.reset_index(drop=True).iterrows():
        roles = set()
        if row["A"]: roles.add("A")
        if row["B"]: roles.add("B")
        if row["C"]: roles.add("C")
        if row["ネコ"]: roles.add("Neko")
        if row["夜可"]: roles.add("Night")
        role_map[i] = roles
    return role_map

def check_night_only(staff_list, role_map, min_night_count):
    return sum(1 for s in staff_list if "Night" in role_map[s]) >= min_night_count

def can_cover_required_roles(staff_list, role_map, min_night_count=3):
    if not check_night_only(staff_list, role_map, min_night_count): return False
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
    
    found_strict = False
    if neko_fixed is not None:
        rem = [x for x in pool if x != neko_fixed]
        for p in itertools.permutations(rem, 3):
            if 'A' in role_map[p[0]] and 'B' in role_map[p[1]] and 'C' in role_map[p[2]]:
                assignments[neko_fixed] = 'ネコ'; assignments[p[0]] = 'A'; assignments[p[1]] = 'B'; assignments[p[2]] = 'C'
                found_strict = True
                for ex in [x for x in rem if x not in p]:
                    caps = role_map[ex]
                    if not any(r in caps for r in ["A","B","C","Neko"]) and "Night" in caps: assignments[ex] = '〇'
                    else: assignments[ex] = 'C' if 'C' in caps else ('B' if 'B' in caps else ('A' if 'A' in caps else 'C'))
                break
    if not found_strict:
        unassigned = set(pool)
        for r in ['A', 'B', 'Neko', 'C']:
            for s in list(unassigned):
                if r == 'Neko' and neko_fixed and neko_fixed in unassigned: assignments[neko_fixed] = 'ネコ'; unassigned.remove(neko_fixed); break
                if r in role_map[s]: assignments[s] = r; unassigned.remove(s); break
        for s in list(unassigned): assignments[s] = '〇' if "Night" in role_map[s] else 'C'
    return assignments

# --- メインロジック ---
def solve_schedule_from_ui(staff_df, holidays_df, days_list, config):
    staff_df = staff_df.dropna(subset=['名前']).reset_index(drop=True)
    num_days, num_staff = len(days_list), len(staff_df)
    role_map = get_role_map_from_df(staff_df)
    min_night = config.get("min_night_staff", 3)
    
    col_prev = "前月末の連勤数" if "前月末の連勤数" in staff_df.columns else "先月からの連勤"
    initial_cons = pd.to_numeric(staff_df[col_prev], errors='coerce').fillna(0).astype(int).values
    req_offs = pd.to_numeric(staff_df['公休数'], errors='coerce').fillna(0).astype(int).values
    max_cons_limits = pd.to_numeric(staff_df['最大連勤'], errors='coerce').fillna(4).astype(int).values
    is_seishain = staff_df['正社員'].astype(bool).values

    fixed_shifts = np.full((num_staff, num_days), '', dtype=object)
    for d_idx in range(num_days):
        col_name = f"Day_{d_idx+1}"
        if col_name in holidays_df.columns:
            for s_idx in range(min(num_staff, len(holidays_df[col_name]))):
                if holidays_df[col_name].values[s_idx] in [True, '×']: fixed_shifts[s_idx, d_idx] = '×'
                    
    day_patterns = []
    for d in range(num_days):
        avail = [s for s in range(num_staff) if fixed_shifts[s, d] != '×']
        pats = [subset for size in range(min(4, len(avail)), len(avail)+1) for subset in itertools.combinations(avail, size)]
        random.shuffle(pats)
        day_patterns.append(pats)
        
    current_paths = [{'sched': np.zeros((num_staff, num_days), dtype=int), 'cons': initial_cons.copy(), 'offs': np.zeros(num_staff, dtype=int), 'score': 0}]
    BEAM_WIDTH = 150
    
    for d in range(num_days):
        next_paths = []
        is_weekend = days_list[d].weekday() >= 5
        for path in current_paths:
            for pat in day_patterns[d][:100]:
                new_cons, new_offs = path['cons'].copy(), path['offs'].copy()
                penalty, violation = 0, False
                
                # 【最優先】夜勤確保ペナルティ（非常に重く）
                if not check_night_only(pat, role_map, min_night): penalty += 200000
                # 次点：日中の役割構成ペナルティ
                if not can_cover_required_roles(pat, role_map, min_night): penalty += 10000
                
                work_mask = np.zeros(num_staff, dtype=int)
                for s in pat: work_mask[s] = 1
                for s in range(num_staff):
                    if work_mask[s]:
                        new_cons[s] += 1
                        if new_cons[s] > max_cons_limits[s]: penalty += 500000
                    else:
                        new_cons[s] = 0; new_offs[s] += 1
                
                days_left = num_days - 1 - d
                for s in range(num_staff):
                    if new_offs[s] > req_offs[s]: penalty += 1000000
                    if new_offs[s] + days_left < req_offs[s]: penalty += 1000000

                next_paths.append({'sched': np.column_stack((path['sched'][:,:d], work_mask)), 'cons': new_cons, 'offs': new_offs, 'score': path['score'] + penalty})
        
        next_paths.sort(key=lambda x: x['score'])
        current_paths = next_paths[:BEAM_WIDTH]
        
    best_path = current_paths[0]
    final_sched = best_path['sched']
    
    # --- 表構築 ---
    weekdays_jp = ["月", "火", "水", "木", "金", "土", "日"]
    multi_cols = pd.MultiIndex.from_arrays([[str(d.day) for d in days_list] + ["勤(休)"], ["祝" if is_holiday(d) else weekdays_jp[d.weekday()] for d in days_list] + [""]])
    output_data = np.full((num_staff + 1, num_days + 1), "", dtype=object)
    
    for d in range(num_days):
        working = [s for s in range(num_staff) if final_sched[s, d] == 1]
        roles = assign_roles_smartly(working, role_map)
        is_insufficient = not can_cover_required_roles(working, role_map, min_night)
        for s in range(num_staff):
            if s in working: output_data[s, d] = roles.get(s, 'C')
            else: output_data[s, d] = '×' if fixed_shifts[s, d] == '×' else '／'
        if is_insufficient: output_data[num_staff, d] = "※"
    
    for s in range(num_staff):
        off_count = sum(1 for x in output_data[s, :num_days] if x in ['／', '×'])
        summary = f"{num_days - off_count}({off_count})"
        if off_count != req_offs[s]: summary += "※"
        output_data[s, num_days] = summary
        
    return pd.DataFrame(output_data, columns=multi_cols, index=list(staff_df['名前']) + ["不足"]), best_path['score']

# --- スタイル設定 ---
def highlight_cells(data):
    styles = pd.DataFrame('', index=data.index, columns=data.columns)
    for col in data.columns:
        if col[1] == '土': styles[col] = 'background-color: #e6f7ff;'
        elif col[1] in ['日', '祝']: styles[col] = 'background-color: #ffe6e6;'
    for r in data.index:
        for c in data.columns:
            val = str(data.at[r, c])
            if c[0] == '勤(休)': styles.at[r, c] += 'font-weight: bold; background-color: #f9f9f9;' + ('color: red;' if "※" in val else '')
            elif val == '／': styles.at[r, c] = 'background-color: #ffcccc;'
            elif val == '×': styles.at[r, c] = 'background-color: #d9d9d9;'
            elif val == '※': styles.at[r, c] = 'background-color: #ff0000; color: white; font-weight: bold;'
            elif val in ['A', 'B', 'C', 'ネコ', '〇']:
                color = {'A': '#ccffff', 'B': '#ccffcc', 'C': '#ffffcc', 'ネコ': '#ffe5cc', '〇': '#e6e6fa'}[val]
                styles.at[r, c] = f'background-color: {color};'
    return styles

# --- UI実装 ---
st.title('📅 シフト作成ツール')
# (サイドバー読込等は前回のまま維持)
# --- メインエリア ---
with st.form("settings_form"):
    with st.expander("🛠 基本設定", expanded=False):
        c1, c2 = st.columns(2)
        new_min_night = c1.number_input("🌙 夜勤最低人数", 1, 10, st.session_state.config.get("min_night_staff", 3))
        st.session_state.config["min_night_staff"] = new_min_night
    
    edited_staff_df = st.data_editor(st.session_state.staff_df, num_rows="dynamic", use_container_width=True, hide_index=True, key="staff_editor")
    # 希望休入力 (前回のコードを流用)
    # ...
    submit_btn = st.form_submit_button("✅ 設定反映")

if st.button("シフトを作成する"):
    with st.spinner("作成中..."):
        result = solve_schedule_from_ui(st.session_state.staff_df, st.session_state.holidays_df, days_list, st.session_state.config)
        if result:
            df, score = result
            st.dataframe(df.style.apply(highlight_cells, axis=None), use_container_width=True, height=600)
            st.download_button("CSV保存", df.to_csv().encode('utf-8-sig'), "shift.csv")
