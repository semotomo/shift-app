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
    /* 名前列 */
    th[aria-label="名前"], td[aria-label="名前"] { max-width: 100px !important; min-width: 100px !important; }
    /* 設定列 */
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
    except ImportError:
        pass
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
        "min_night_staff": 3,          
        "enable_seishain_rule": True,  
        "priority_days": ["土", "日"],  
        "consecutive_penalty_weight": "通常" 
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
                    if col == "最大連勤": staff_df[col] = 4
                    elif col == "正社員": staff_df[col] = False
                    elif col == "朝可": staff_df[col] = True
                    else: staff_df[col] = False
            
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
    if today.day >= 26: start_date = today.replace(day=26)
    else: start_date = today.replace(day=26)
    if start_date.month == 12: end_date = start_date.replace(year=start_date.year + 1, month=1, day=25)
    else: end_date = start_date.replace(month=start_date.month + 1, day=25)
    return start_date, end_date

def get_default_data():
    staff_data = {
        "名前": ["西原", "松本", "中島", "山下", "下尾", "原", "松尾"],
        "正社員": [True, True, True, True, False, False, False],
        "朝可": [True, True, True, True, False, True, True],
        "夜可": [True, True, True, True, True, False, False], 
        "A": [True, True, False, False, False, False, False],
        "B": [False, True, True, True, False, False, False],
        "C": [False, False, True, True, False, True, True],
        "ネコ": [False, True, True, True, False, True, True],
        "前月末の連勤数": [0, 5, 1, 0, 0, 2, 2],
        "最大連勤": [4, 4, 4, 4, 3, 4, 3],
        "公休数": [8, 8, 8, 8, 13, 9, 15]
    }
    holidays_data = pd.DataFrame(False, index=range(7), columns=[f"Day_{i+1}" for i in range(31)])
    return pd.DataFrame(staff_data), holidays_data

if 'staff_df' not in st.session_state:
    loaded_staff, loaded_holidays, l_start, l_end, l_config = load_settings_from_file()
    if loaded_staff is not None:
        st.session_state.staff_df = loaded_staff
        st.session_state.holidays_df = loaded_holidays
        st.session_state.loaded_start_date = l_start
        st.session_state.loaded_end_date = l_end
        st.session_state.config = l_config if l_config else get_default_config()
    else:
        d_staff, d_holidays = get_default_data()
        st.session_state.staff_df = d_staff
        st.session_state.holidays_df = d_holidays
        st.session_state.loaded_start_date = None
        st.session_state.loaded_end_date = None
        st.session_state.config = get_default_config()

# --- ロジック関数 ---
def get_role_map_from_df(staff_df):
    role_map = {}
    df = staff_df.reset_index(drop=True)
    for i, row in df.iterrows():
        roles = set()
        if row["A"]: roles.add("A")
        if row["B"]: roles.add("B")
        if row["C"]: roles.add("C")
        if row["ネコ"]: roles.add("Neko")
        if row["夜可"]: roles.add("Night")
        role_map[i] = roles
    return role_map

def can_cover_required_roles(staff_list, role_map, min_night_count=3):
    # 夜勤の人数確保を最優先判定
    if sum(1 for s in staff_list if "Night" in role_map[s]) < min_night_count: return False
    
    neko_cands = [s for s in staff_list if "Neko" in role_map[s]]
    p_neko = [s for s in neko_cands if "A" not in role_map[s] and "B" not in role_map[s]]
    neko_fixed = p_neko[0] if p_neko else (neko_cands[0] if neko_cands else None)
    
    if neko_fixed is not None:
        rem = [x for x in staff_list if x != neko_fixed]
        if len(rem) < 3: return False
        if not all(any(r in role_map[x] for x in rem) for r in ["A", "B", "C"]): return False
        for p in itertools.permutations(rem, 3):
             if 'A' in role_map[p[0]] and 'B' in role_map[p[1]] and 'C' in role_map[p[2]]: return True
        return False
    else:
        if len(staff_list) < 4: return False
        for p in itertools.permutations(staff_list, 4):
             if 'Neko' in role_map[p[0]] and 'A' in role_map[p[1]] and 'B' in role_map[p[2]] and 'C' in role_map[p[3]]: return True
        return False

def get_possible_day_patterns(available_staff):
    n = len(available_staff)
    if n < 4: return [tuple(available_staff)] 
    return [subset for size in range(4, min(n+1, 10)) for subset in itertools.combinations(available_staff, size)]

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
                    if not any(r in role_map[ex] for r in ["A","B","C","Neko"]) and "Night" in role_map[ex]: assignments[ex] = '〇'
                    else:
                        caps = role_map[ex]
                        output_role = 'C'
                        if 'C' in caps: output_role = 'C'
                        elif 'B' in caps: output_role = 'B'
                        elif 'A' in caps: output_role = 'A'
                        elif 'Neko' in caps: output_role = 'ネコ'
                        if "Night" in role_map[ex] and not any(r in role_map[ex] for r in ["A","B","C","Neko"]): output_role = '〇'
                        assignments[ex] = output_role
                break
    else:
        for p in itertools.permutations(pool, 4):
            if 'Neko' in role_map[p[0]] and 'A' in role_map[p[1]] and 'B' in role_map[p[2]] and 'C' in role_map[p[3]]:
                assignments[p[0]] = 'ネコ'; assignments[p[1]] = 'A'; assignments[p[2]] = 'B'; assignments[p[3]] = 'C'
                found_strict = True
                for ex in [x for x in pool if x not in p]:
                    if not any(r in role_map[ex] for r in ["A","B","C","Neko"]) and "Night" in role_map[ex]: assignments[ex] = '〇'
                    else:
                        caps = role_map[ex]
                        output_role = 'C'
                        if 'C' in caps: output_role = 'C'
                        elif 'B' in caps: output_role = 'B'
                        elif 'A' in caps: output_role = 'A'
                        assignments[ex] = output_role
                break
    if not found_strict:
        unassigned = set(pool)
        for r in ['A', 'B', 'Neko', 'C']:
            for s in list(unassigned):
                if r == 'Neko' and neko_fixed and neko_fixed in unassigned: assignments[neko_fixed] = 'ネコ'; unassigned.remove(neko_fixed); break
                if r in role_map[s]: assignments[s] = r; unassigned.remove(s); break
        for s in list(unassigned):
            if "Night" in role_map[s] and not any(r in role_map[s] for r in ["A","B","C","Neko"]): assignments[s] = '〇'
            elif 'C' in role_map[s]: assignments[s] = 'C'
    return assignments

# --- メインロジック ---
def solve_schedule_from_ui(staff_df, holidays_df, days_list, config):
    staff_df = staff_df.dropna(subset=['名前'])
    staff_df = staff_df[staff_df['名前'] != '']
    staff_df = staff_df.reset_index(drop=True)
    num_days = len(days_list)
    num_staff = len(staff_df)
    if num_staff == 0: return None
    role_map = get_role_map_from_df(staff_df)
    
    min_night = config.get("min_night_staff", 3)
    enable_seishain_rule = config.get("enable_seishain_rule", True)
    priority_days = config.get("priority_days", ["土", "日"])
    penalty_weight = config.get("consecutive_penalty_weight", "通常")
    cons_penalty_factor = 2000 if penalty_weight == "厳格" else (1000 if penalty_weight == "通常" else 500)
    
    col_prev_cons = "前月末の連勤数" if "前月末の連勤数" in staff_df.columns else "先月からの連勤"
    initial_cons = pd.to_numeric(staff_df[col_prev_cons], errors='coerce').fillna(0).astype(int).values
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
        pats = get_possible_day_patterns(avail)
        random.shuffle(pats)
        day_patterns.append(pats)
        
    current_paths = [{
        'sched': np.zeros((num_staff, num_days), dtype=int), 
        'cons': initial_cons.copy(), 
        'offs': np.zeros(num_staff, dtype=int), 
        'off_cons': np.zeros(num_staff, dtype=int), 
        'weekend_offs': np.zeros(num_staff, dtype=int),
        'score': 0
    }]
    
    BEAM_WIDTH = 300
    weekdays_jp = ["月", "火", "水", "木", "金", "土", "日"]
    
    for d in range(num_days):
        day_obj = days_list[d]
        is_weekend = day_obj.weekday() >= 5
        day_str = weekdays_jp[day_obj.weekday()]
        is_priority_day = day_str in priority_days
        
        next_paths = []
        patterns = day_patterns[d]
        
        valid_pats = [p for p in patterns if can_cover_required_roles(p, role_map, min_night)]
        invalid_pats = [p for p in patterns if not can_cover_required_roles(p, role_map, min_night)]
        use_patterns = valid_pats[:200] + invalid_pats[:50]
        
        for path in current_paths:
            for pat in use_patterns:
                new_cons = path['cons'].copy()
                new_offs = path['offs'].copy()
                new_off_cons = path['off_cons'].copy()
                new_weekend_offs = path['weekend_offs'].copy()
                penalty = 0
                
                # 【重要】夜勤の人員確保不足に超巨大ペナルティ（休日数遵守と同レベル）
                if not can_cover_required_roles(pat, role_map, min_night):
                    penalty += 800000 
                
                staff_count = len(pat)
                if is_priority_day and staff_count <= 4: penalty += 30 

                work_mask = np.zeros(num_staff, dtype=int)
                for s in pat: work_mask[s] = 1
                
                for s in range(num_staff):
                    limit = max_cons_limits[s]
                    if work_mask[s] == 1:
                        new_cons[s] += 1; new_off_cons[s] = 0
                        if new_cons[s] > limit:
                            if new_cons[s] == limit + 1: penalty += cons_penalty_factor
                            else: penalty += 500000 
                        elif new_cons[s] == limit: penalty += 50
                    else:
                        new_cons[s] = 0; new_offs[s] += 1; new_off_cons[s] += 1
                        if enable_seishain_rule and is_weekend and is_seishain[s]:
                            new_weekend_offs[s] += 1
                            if new_weekend_offs[s] > 2: penalty += 50
                        if new_off_cons[s] >= 3:
                            penalty += 100
                            if "Neko" in role_map[s] and "C" in role_map[s] and "A" not in role_map[s]: penalty += 200
                
                days_left = num_days - 1 - d
                for s in range(num_staff):
                    if new_offs[s] > req_offs[s]: penalty += 1000000 
                    if new_offs[s] + days_left < req_offs[s]: penalty += 1000000 

                expected = req_offs * ((d+1)/num_days)
                penalty += np.sum(np.abs(new_offs - expected)) * 10
                
                new_sched = path['sched'].copy(); new_sched[:, d] = work_mask
                next_paths.append({
                    'sched': new_sched, 'cons': new_cons, 'offs': new_offs, 
                    'off_cons': new_off_cons, 'weekend_offs': new_weekend_offs, 'score': path['score'] + penalty
                })
        
        next_paths.sort(key=lambda x: x['score'])
        if not next_paths: return None
        current_paths = next_paths[:BEAM_WIDTH]
        
    best_path = current_paths[0]
    final_sched = best_path['sched']
    final_score = best_path['score']
    
    # --- 完成シフト表構築 ---
    top_level = [str(d.day) for d in days_list] + ["勤(休)"]
    bottom_level = ["祝" if is_holiday(d) else weekdays_jp[d.weekday()] for d in days_list] + [""]
    multi_cols = pd.MultiIndex.from_arrays([top_level, bottom_level])
    output_data = np.full((num_staff + 1, num_days + 1), "", dtype=object)
    
    for d in range(num_days):
        working = [s for s in range(num_staff) if final_sched[s, d] == 1]
        roles = assign_roles_smartly(working, role_map)
        is_insufficient = not can_cover_required_roles(working, role_map, min_night)
        
        for s in range(num_staff):
            if s in working:
                if s in roles: output_data[s, d] = roles[s]
                else:
                    caps = role_map[s]
                    output_data[s, d] = 'C' if 'C' in caps else ('B' if 'B' in caps else ('A' if 'A' in caps else 'C'))
            else: 
                output_data[s, d] = '×' if fixed_shifts[s, d] == '×' else '／'
        
        if is_insufficient: output_data[num_staff, d] = "※"
    
    for s in range(num_staff):
        shifts = output_data[s, :num_days]
        off_count = sum(1 for x in shifts if x in ['／', '×'])
        work_count = num_days - off_count
        summary_str = f"{work_count}({off_count})"
        if off_count != req_offs[s]:
             summary_str += "※"
        output_data[s, num_days] = summary_str
        
    output_data[num_staff, num_days] = ""
    index_names = list(staff_df['名前']) + ["不足"]
    return pd.DataFrame(output_data, columns=multi_cols, index=index_names), final_score

# --- CSV生成 ---
def generate_custom_csv(result_df, staff_df, days_list):
    weekdays_jp = ["月", "火", "水", "木", "金", "土", "日"]
    row1 = ["", "本店"]
    current_m = days_list[0].month
    count = 0
    for d in days_list:
        if d.month == current_m:
            row1.append(f"　{current_m}月 " if count == 0 else "")
            count += 1
        else:
            current_m = d.month
            count = 1
            row1.append(f"　{current_m}月 ")
    row1.append("")
    row2 = ["", "日にち"] + [str(d.day) for d in days_list] + ["勤(休)"]
    row3 = ["\"先月からの\n連勤日数\"", "曜日"]
    for d in days_list:
        row3.append("祝" if is_holiday(d) else weekdays_jp[d.weekday()])
    row3.append("")
    data_rows = []
    col_prev_cons = "前月末の連勤数" if "前月末の連勤数" in staff_df.columns else "先月からの連勤"
    prev_cons_map = {row['名前']: row[col_prev_cons] for _, row in staff_df.iterrows()}
    for name, row in result_df.iterrows():
        if name == "不足": continue
        p_cons = prev_cons_map.get(name, 0)
        data_rows.append([str(p_cons), name] + list(row.values))
    lines = [",".join(row1), ",".join(row2), ",".join(row3)]
    for dr in data_rows: lines.append(",".join([str(x) for x in dr]))
    return "\n".join(lines).encode('utf-8-sig')

# --- スタイル設定 ---
def highlight_cells(data):
    styles = pd.DataFrame('', index=data.index, columns=data.columns)
    for col in data.columns:
        week_str = col[1]
        if week_str == '土': styles[col] = 'background-color: #e6f7ff;'
        elif week_str in ['日', '祝']: styles[col] = 'background-color: #ffe6e6;'
    for r in data.index:
        for c in data.columns:
            val = str(data.at[r, c])
            if c[0] == '勤(休)':
                styles.at[r, c] += 'font-weight: bold; background-color: #f9f9f9;'
                if "※" in val: styles.at[r, c] += 'color: red;'
                continue
            
            if val == '／': styles.at[r, c] = 'background-color: #ffcccc; color: black;'
            elif val == '×': styles.at[r, c] = 'background-color: #d9d9d9; color: gray;'
            elif val == '※': styles.at[r, c] = 'background-color: #ff0000; color: white; font-weight: bold;'
            elif val == 'A': styles.at[r, c] = 'background-color: #ccffff; color: black;'
            elif val == 'B': styles.at[r, c] = 'background-color: #ccffcc; color: black;'
            elif val == 'C': styles.at[r, c] = 'background-color: #ffffcc; color: black;'
            elif val == 'ネコ': styles.at[r, c] = 'background-color: #ffe5cc; color: black;'
            elif val == '〇': styles.at[r, c] = 'background-color: #e6e6fa; color: black;'
    return styles

# ==========================================
# UI実装
# ==========================================
st.title('📅 シフト作成ツール')

with st.sidebar:
    st.header("⚙️ 保存・読込")
    save_clicked = st.button("💾 設定をサーバーに保存", type="primary")

    st.markdown("---")
    st.header("📅 日付設定")
    default_start, default_end = get_default_date_range()
    if st.session_state.loaded_start_date: default_start = st.session_state.loaded_start_date
    if st.session_state.loaded_end_date: default_end = st.session_state.loaded_end_date

    col_d1, col_d2 = st.columns(2)
    start_input = col_d1.date_input("開始日", default_start, format="YYYY/MM/DD")
    end_input = col_d2.date_input("終了日", default_end, format="YYYY/MM/DD")
    days_list = pd.date_range(start_input, end_input).tolist()
    num_days = len(days_list)
    
    if save_clicked:
        clean_staff_df = st.session_state.staff_df.dropna(subset=['名前'])
        clean_staff_df = clean_staff_df[clean_staff_df['名前'] != '']
        save_dict = {
            "staff": clean_staff_df.to_dict(),
            "holidays": st.session_state.holidays_df.to_dict(),
            "date_range": {"start": start_input.strftime("%Y-%m-%d"), "end": end_input.strftime("%Y-%m-%d")},
            "config": st.session_state.config 
        }
        try:
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(save_dict, f, ensure_ascii=False, indent=2)
            st.success("保存しました！")
        except Exception as e: st.error(f"保存失敗: {e}")

    st.markdown("---")
    st.subheader("📥 バックアップ")
    clean_staff_df = st.session_state.staff_df.dropna(subset=['名前'])
    json_str = json.dumps({
        "staff": clean_staff_df.to_dict(),
        "holidays": st.session_state.holidays_df.to_dict(),
        "date_range": {"start": start_input.strftime("%Y-%m-%d"), "end": end_input.strftime("%Y-%m-%d")},
        "config": st.session_state.config
    }, ensure_ascii=False)
    st.download_button("設定ファイルDL", json_str, "shift_settings.json", "application/json")
    
    uploaded_json = st.file_uploader("設定ファイル読込", type=["json"])
    if uploaded_json is not None:
        try:
            loaded_data = json.load(uploaded_json)
            df_new = pd.DataFrame(loaded_data["staff"])
            for col in ["正社員", "朝可", "夜可", "A", "B", "C", "ネコ", "最大連勤"]:
                if col not in df_new.columns:
                    if col == "最大連勤": df_new[col] = 4
                    elif col == "正社員": df_new[col] = False
                    elif col == "朝可": df_new[col] = True
                    else: df_new[col] = False
            if "先月からの連勤" in df_new.columns: df_new["前月末の連勤数"] = df_new["先月からの連勤"]
            st.session_state.staff_df = df_new
            st.session_state.holidays_df = pd.DataFrame(loaded_data["holidays"])
            if "date_range" in loaded_data:
                st.session_state.loaded_start_date = datetime.datetime.strptime(loaded_data["date_range"]["start"], "%Y-%m-%d").date()
                st.session_state.loaded_end_date = datetime.datetime.strptime(loaded_data["date_range"]["end"], "%Y-%m-%d").date()
            if "config" in loaded_data:
                st.session_state.config = loaded_data["config"]
            st.rerun()
        except: st.error("読込エラー")

# --- メインエリア ---
with st.form("settings_form"):
    with st.expander("🛠 基本設定（クリックで開閉）", expanded=False):
        c_conf1, c_conf2 = st.columns(2)
        with c_conf1:
            st.markdown("##### 👥 人数・ルール設定")
            new_min_night = st.number_input("🌙 夜勤の最低必要人数", min_value=1, max_value=10, value=st.session_state.config.get("min_night_staff", 3))
            new_seishain_rule = st.checkbox("👔 正社員の土日休み制限（月1回推奨）を有効にする", value=st.session_state.config.get("enable_seishain_rule", True))
            new_penalty = st.selectbox("⚠️ 最大連勤ペナルティの重さ", ["通常", "厳格", "緩め"], index=["通常", "厳格", "緩め"].index(st.session_state.config.get("consecutive_penalty_weight", "通常")))
        
        with c_conf2:
            st.markdown("##### 🗓 曜日優先度")
            st.caption("ここで選んだ曜日は、優先的に人員を確保します")
            weekdays = ["月", "火", "水", "木", "金", "土", "日"]
            default_priority = st.session_state.config.get("priority_days", ["土", "日"])
            new_priority_days = st.multiselect("優先確保する曜日", weekdays, default=default_priority)
        
        st.session_state.config.update({
            "min_night_staff": new_min_night,
            "enable_seishain_rule": new_seishain_rule,
            "priority_days": new_priority_days,
            "consecutive_penalty_weight": new_penalty
        })

    st.markdown("### 1️⃣ スタッフ設定")
    st.info("💡 変更後、下の **「✅ 設定を反映して保存」** ボタンを押してください。")
    
    edited_staff_df = st.data_editor(
        st.session_state.staff_df, num_rows="dynamic", use_container_width=True, hide_index=True, key="staff_editor",
        column_config={
            "正社員": st.column_config.CheckboxColumn("社員", width="small", default=False),
            "朝可": st.column_config.CheckboxColumn("朝", width="small", default=True),
            "夜可": st.column_config.CheckboxColumn("夜", width="small", default=False),
            "A": st.column_config.CheckboxColumn("A", width="small", default=False),
            "B": st.column_config.CheckboxColumn("B", width="small", default=False),
            "C": st.column_config.CheckboxColumn("C", width="small", default=False),
            "ネコ": st.column_config.CheckboxColumn("🐱", width="small", default=False),
            "前月末の連勤数": st.column_config.NumberColumn("前連勤", width="small"),
            "最大連勤": st.column_config.NumberColumn("MAX連", width="small", default=4),
            "公休数": st.column_config.NumberColumn("公休", width="small"),
            "名前": st.column_config.TextColumn("名前", width="small"),
        }
    )
    
    st.markdown("### 2️⃣ 希望休入力")
    holiday_cols = [f"Day_{i+1}" for i in range(num_days)]
    display_holidays_df = st.session_state.holidays_df.copy().reindex(columns=holiday_cols, fill_value=False)
    
    weekdays_jp = ["月", "火", "水", "木", "金", "土", "日"]
    ui_cols = ["名前"]
    for d in days_list:
        week_str = "祝" if is_holiday(d) else weekdays_jp[d.weekday()]
        ui_cols.append(f"{d.day}\n{week_str}")
    
    if len(display_holidays_df) == len(st.session_state.staff_df):
        display_holidays_df.insert(0, "名前", st.session_state.staff_df['名前'].values)
    else:
        display_holidays_df.insert(0, "名前", [""] * len(display_holidays_df))
        
    display_holidays_df.columns = ui_cols
    col_config_holidays = {"名前": st.column_config.TextColumn("名前", disabled=True, width="small")}
    for i in range(len(days_list)): col_config_holidays[ui_cols[i+1]] = st.column_config.CheckboxColumn(width="small", default=False)

    edited_holidays_grid = st.data_editor(display_holidays_df, use_container_width=True, hide_index=True, key="holidays_editor", column_config=col_config_holidays)
    submit_btn = st.form_submit_button("✅ 設定を反映して保存", type="primary")

if submit_btn:
    st.session_state.staff_df = edited_staff_df
    valid_staff_count = len(edited_staff_df[edited_staff_df['名前'].notna() & (edited_staff_df['名前'] != "")])
    new_holidays = edited_holidays_grid.drop(columns=["名前"])
    new_holidays.columns = holiday_cols 
    if valid_staff_count > len(new_holidays):
        new_holidays = pd.concat([new_holidays, pd.DataFrame(False, index=range(valid_staff_count - len(new_holidays)), columns=new_holidays.columns)], ignore_index=True)
    elif valid_staff_count < len(new_holidays):
        new_holidays = new_holidays.iloc[:valid_staff_count]
    st.session_state.holidays_df = new_holidays
    st.success("設定を更新しました！")
    st.rerun()

st.markdown("### 3️⃣ シフト作成")
if st.button("シフトを作成する"):
    with st.spinner("AIが最優先で夜勤枠を埋めています...🧩"):
        try:
            result = solve_schedule_from_ui(st.session_state.staff_df, st.session_state.holidays_df, days_list, st.session_state.config)
            if result is not None:
                result_df, final_score = result
                
                if final_score >= 1000000:
                    st.warning("⚠️ 【AIからの報告】どうしても公休数を守れないスタッフがいました。")
                elif final_score >= 800000:
                    st.warning("⚠️ 【AIからの報告】最優先しましたが、夜勤人数が足りない日があります。（不足行の※を確認してください）")
                else:
                    st.success("✨ 作成完了！夜勤を含め、条件を綺麗に満たしたシフトができました。")

                st.subheader(f"{days_list[0].month}月度 シフト表")
                styled_df = result_df.style.apply(highlight_cells, axis=None)
                st.dataframe(styled_df, use_container_width=True, height=600)
                
                csv_data = generate_custom_csv(result_df, st.session_state.staff_df, days_list)
                st.download_button("📥 CSVダウンロード (エクセル対応)", csv_data, "shift_result.csv", "text/csv")
            else:
                st.error("システムエラーで作成できませんでした。")
        except Exception as e:
            st.error(f"エラー: {e}")
