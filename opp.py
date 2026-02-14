import streamlit as st
import pandas as pd
import numpy as np
import random
import itertools
import json
import datetime
import os

# --- ページ設定 ---
st.set_page_config(page_title="シフト作成ツール(安定版)", layout="wide")

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
            cols_def = {"正社員": False, "朝可": True, "夜可": False, "A": False, "B": False, "C": False, "ネコ": False, "最大連勤": 4, "レベル": "スタッフ"}
            for col, val in cols_def.items():
                if col not in staff_df.columns: staff_df[col] = val
            start_d, end_d = None, None
            if "date_range" in loaded_data:
                try:
                    start_d = datetime.datetime.strptime(loaded_data["date_range"]["start"], "%Y-%m-%d").date()
                    end_d = datetime.datetime.strptime(loaded_data["date_range"]["end"], "%Y-%m-%d").date()
                except: pass
            config = loaded_data.get("config", get_default_config())
            pairs_data = loaded_data.get("pairs", [])
            pairs_df = pd.DataFrame(pairs_data)
            if pairs_df.empty: pairs_df = pd.DataFrame(columns=["Staff A", "Staff B", "Type"])
            return staff_df, pd.DataFrame(loaded_data["holidays"]), start_d, end_d, config, pairs_df
        except Exception: return None, None, None, None, None, None
    return None, None, None, None, None, None

def get_default_date_range():
    today = datetime.date.today()
    start_date = today.replace(day=26)
    if start_date.month == 12: end_date = start_date.replace(year=start_date.year + 1, month=1, day=25)
    else: end_date = start_date.replace(month=start_date.month + 1, day=25)
    return start_date, end_date

# --- ロジック関数 ---
def get_role_map_from_df(staff_df):
    role_map, level_map = {}, {}
    df = staff_df.reset_index(drop=True)
    for i, row in df.iterrows():
        roles = set()
        if row["A"]: roles.add("A")
        if row["B"]: roles.add("B")
        if row["C"]: roles.add("C")
        if row["ネコ"]: roles.add("Neko")
        if row["夜可"]: roles.add("Night")
        role_map[i] = roles
        level_map[i] = row["レベル"]
    return role_map, level_map

def can_cover_required_roles(staff_list, role_map, level_map, min_night_count=3):
    if sum(1 for s in staff_list if "Night" in role_map[s]) < min_night_count: return False
    if sum(1 for s in staff_list if level_map[s] == "リーダー") < 1: return False
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
                    caps = role_map[ex]
                    output_role = 'C' if 'C' in caps else ('B' if 'B' in caps else ('A' if 'A' in caps else ('ネコ' if 'Neko' in caps else '〇')))
                    if "Night" in role_map[ex] and not any(r in role_map[ex] for r in ["A","B","C","Neko"]): output_role = '〇'
                    assignments[ex] = output_role
                break
    else:
        for p in itertools.permutations(pool, 4):
            if 'Neko' in role_map[p[0]] and 'A' in role_map[p[1]] and 'B' in role_map[p[2]] and 'C' in role_map[p[3]]:
                assignments[p[0]] = 'ネコ'; assignments[p[1]] = 'A'; assignments[p[2]] = 'B'; assignments[p[3]] = 'C'
                found_strict = True
                for ex in [x for x in pool if x not in p]:
                    caps = role_map[ex]
                    assignments[ex] = 'C' if 'C' in caps else ('B' if 'B' in caps else ('A' if 'A' in caps else '〇'))
                break
    if not found_strict:
        unassigned = set(pool)
        for r in ['A', 'B', 'Neko', 'C']:
            for s in list(unassigned):
                if r == 'Neko' and neko_fixed and neko_fixed in unassigned: assignments[neko_fixed] = 'ネコ'; unassigned.remove(neko_fixed); break
                if r in role_map[s]: assignments[s] = r; unassigned.remove(s); break
        for s in list(unassigned): assignments[s] = '〇'
    return assignments

# --- メインロジック ---
def solve_schedule_from_ui(staff_df, holidays_df, days_list, config, pairs_df):
    staff_df = staff_df.dropna(subset=['名前']).reset_index(drop=True)
    num_days, num_staff = len(days_list), len(staff_df)
    role_map, level_map = get_role_map_from_df(staff_df)
    name_to_idx = {name: i for i, name in enumerate(staff_df['名前'])}
    pair_constraints = []
    if not pairs_df.empty:
        for _, row in pairs_df.iterrows():
            if row.get("Staff A") in name_to_idx and row.get("Staff B") in name_to_idx:
                pair_constraints.append({"a": name_to_idx[row["Staff A"]], "b": name_to_idx[row["Staff B"]], "type": row["Type"]})

    min_night = config.get("min_night_staff", 3)
    priority_days = config.get("priority_days", ["土", "日"])
    penalty_weight = config.get("consecutive_penalty_weight", "通常")
    cons_penalty_factor = 2000 if penalty_weight == "厳格" else (1000 if penalty_weight == "通常" else 500)
    
    col_prev_cons = "前月末の連勤数" if "前月末の連勤数" in staff_df.columns else "先月からの連勤"
    initial_cons = pd.to_numeric(staff_df[col_prev_cons], errors='coerce').fillna(0).astype(int).values
    req_offs = pd.to_numeric(staff_df['公休数'], errors='coerce').fillna(0).astype(int).values
    max_cons_limits = pd.to_numeric(staff_df['最大連勤'], errors='coerce').fillna(4).astype(int).values
    
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
        day_patterns.append(pats[:500]) 
        
    current_paths = [{
        'sched': np.zeros((num_staff, num_days), dtype=int), 
        'cons': initial_cons.copy(), 
        'offs': np.zeros(num_staff, dtype=int), 
        'off_cons': np.zeros(num_staff, dtype=int), 
        'score': 0
    }]
    
    BEAM_WIDTH = 300
    for d in range(num_days):
        is_priority_day = ["月", "火", "水", "木", "金", "土", "日"][days_list[d].weekday()] in priority_days
        next_paths = []
        for path in current_paths:
            for pat in day_patterns[d]:
                penalty = 0
                # 【最重要：公休死守】公休数を超えて休む・働くことへの超特大ペナルティ
                new_offs = path['offs'].copy()
                days_left = num_days - 1 - d
                work_mask = np.zeros(num_staff, dtype=int)
                for s in pat: work_mask[s] = 1
                
                for s in range(num_staff):
                    if work_mask[s] == 0: new_offs[s] += 1
                    if new_offs[s] > req_offs[s]: penalty += 10000000 # 休みすぎNG
                    if new_offs[s] + days_left < req_offs[s]: penalty += 10000000 # 休み不足NG

                # 【次点：役割要件】夜勤、リーダー、ペア
                if not can_cover_required_roles(pat, role_map, level_map, min_night):
                    penalty += 500000 # 公休死守よりは低いが、高いペナルティ
                
                for const in pair_constraints:
                    a_in, b_in = (const["a"] in pat), (const["b"] in pat)
                    if const["type"] == "NG" and a_in and b_in: penalty += 100000
                    elif const["type"] == "Pair" and (a_in != b_in): penalty += 100000

                if is_priority_day and len(pat) <= 4: penalty += 100
                
                new_cons = path['cons'].copy()
                for s in range(num_staff):
                    if work_mask[s] == 1:
                        new_cons[s] += 1
                        if new_cons[s] > max_cons_limits[s]: penalty += cons_penalty_factor
                    else: new_cons[s] = 0
                
                penalty += np.sum(np.abs(new_offs - req_offs * ((d+1)/num_days))) * 10
                new_sched = path['sched'].copy(); new_sched[:, d] = work_mask
                next_paths.append({'sched': new_sched, 'cons': new_cons, 'offs': new_offs, 'score': path['score'] + penalty})
        
        next_paths.sort(key=lambda x: x['score'])
        if not next_paths: return None
        current_paths = next_paths[:BEAM_WIDTH]
        
    best_path = current_paths[0]
    final_sched = best_path['sched']
    output_data = np.full((num_staff + 1, num_days + 1), "", dtype=object)
    
    for d in range(num_days):
        working = [s for s in range(num_staff) if final_sched[s, d] == 1]
        roles = assign_roles_smartly(working, role_map)
        is_insufficient = not can_cover_required_roles(working, role_map, level_map, min_night)
        for s in range(num_staff):
            if s in working: output_data[s, d] = roles.get(s, '〇')
            else: output_data[s, d] = '×' if fixed_shifts[s, d] == '×' else '／'
        if is_insufficient: output_data[num_staff, d] = "※"
    
    for s in range(num_staff):
        off_count = sum(1 for x in output_data[s, :num_days] if x in ['／', '×'])
        output_data[s, num_days] = f"{num_days - off_count}({off_count})" + ("※" if off_count != req_offs[s] else "")
    
    index_names = list(staff_df['名前']) + ["不足"]
    weekdays_jp = ["月", "火", "水", "木", "金", "土", "日"]
    multi_cols = pd.MultiIndex.from_arrays([[str(d.day) for d in days_list] + ["勤(休)"], ["祝" if is_holiday(d) else weekdays_jp[d.weekday()] for d in days_list] + [""]])
    return pd.DataFrame(output_data, columns=multi_cols, index=index_names), best_path['score']

# --- 以下UI/保存読込（以前の機能を維持） ---
if 'staff_df' not in st.session_state:
    l_staff, l_holidays, l_start, l_end, l_config, l_pairs = load_settings_from_file()
    if l_staff is not None:
        st.session_state.staff_df, st.session_state.holidays_df, st.session_state.config, st.session_state.pairs_df = l_staff, l_holidays, l_config, l_pairs
        st.session_state.loaded_start_date, st.session_state.loaded_end_date = l_start, l_end
    else:
        st.session_state.staff_df, st.session_state.holidays_df, st.session_state.pairs_df = get_default_data()
        st.session_state.config = get_default_config()
        st.session_state.loaded_start_date, st.session_state.loaded_end_date = None, None

# --- UI実装 ---
st.title('📅 シフト作成ツール')

with st.sidebar:
    st.header("⚙️ 保存・読込")
    save_clicked = st.button("💾 設定をサーバーに保存", type="primary")
    st.header("📅 日付設定")
    d_start, d_end = get_default_date_range()
    if st.session_state.loaded_start_date: d_start = st.session_state.loaded_start_date
    if st.session_state.loaded_end_date: d_end = st.session_state.loaded_end_date
    start_input = st.date_input("開始日", d_start, format="YYYY/MM/DD")
    end_input = st.date_input("終了日", d_end, format="YYYY/MM/DD")
    days_list = pd.date_range(start_input, end_input).tolist()
    
    if save_clicked:
        try:
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump({"staff": st.session_state.staff_df.to_dict(), "holidays": st.session_state.holidays_df.to_dict(), "date_range": {"start": start_input.strftime("%Y-%m-%d"), "end": end_input.strftime("%Y-%m-%d")}, "config": st.session_state.config, "pairs": st.session_state.pairs_df.to_dict()}, f, ensure_ascii=False, indent=2)
            st.success("保存完了")
        except Exception as e: st.error(f"エラー: {e}")

# --- メインエリア：設定フォーム ---
with st.form("settings_form"):
    with st.expander("🛠 基本設定・ペア設定"):
        c1, c2 = st.columns(2)
        st.session_state.config["min_night_staff"] = c1.number_input("🌙 夜勤最低人数", 1, 10, st.session_state.config["min_night_staff"])
        st.session_state.config["consecutive_penalty_weight"] = c2.selectbox("⚠️ 連勤制限の強さ", ["通常", "厳格", "緩め"], index=["通常", "厳格", "緩め"].index(st.session_state.config["consecutive_penalty_weight"]))
        st.session_state.pairs_df = st.data_editor(st.session_state.pairs_df, num_rows="dynamic", use_container_width=True, column_config={"Staff A": st.column_config.SelectboxColumn("スタッフ A", options=st.session_state.staff_df['名前'].unique()), "Staff B": st.column_config.SelectboxColumn("スタッフ B", options=st.session_state.staff_df['名前'].unique()), "Type": st.column_config.SelectboxColumn("タイプ", options=["NG", "Pair"])})

    st.markdown("### 1️⃣ スタッフ設定")
    st.session_state.staff_df = st.data_editor(st.session_state.staff_df, num_rows="dynamic", use_container_width=True, hide_index=True)
    st.markdown("### 2️⃣ 希望休入力")
    display_holidays_df = st.session_state.holidays_df.copy().reindex(columns=[f"Day_{i+1}" for i in range(len(days_list))], fill_value=False)
    display_holidays_df.insert(0, "名前", st.session_state.staff_df['名前'].values[:len(display_holidays_df)])
    edited_holidays = st.data_editor(display_holidays_df, use_container_width=True, hide_index=True)
    submit_btn = st.form_submit_button("✅ 設定を反映して保存", type="primary")

if submit_btn:
    st.session_state.holidays_df = edited_holidays.drop(columns=["名前"])
    st.success("更新しました。")
    st.rerun()

st.markdown("### 3️⃣ シフト作成")
if st.button("シフトを作成する"):
    with st.spinner("シフトを作成中..."):
        res = solve_schedule_from_ui(st.session_state.staff_df, st.session_state.holidays_df, days_list, st.session_state.config, st.session_state.pairs_df)
        if res:
            df, score = res
            if score >= 10000000: st.error("⚠️ 希望休が多すぎる等の理由で、指定された公休数を守れませんでした。スタッフの休みを調整してください。")
            elif score >= 500000: st.warning("⚠️ 公休数は守りましたが、人員不足の日があります（※マーク）。")
            else: st.success("✨ 完璧なシフトが作成できました！")
            
            # --- スタイル設定 ---
            def highlight(data):
                styles = pd.DataFrame('', index=data.index, columns=data.columns)
                for r in data.index:
                    for c in data.columns:
                        val = str(data.at[r, c])
                        if val == '／': styles.at[r, c] = 'background-color: #ffcccc'
                        elif val == '×': styles.at[r, c] = 'background-color: #d9d9d9'
                        elif val == '※': styles.at[r, c] = 'background-color: #ff0000; color: white'
                return styles
            st.dataframe(df.style.apply(highlight, axis=None), use_container_width=True, height=600)
