import streamlit as st

import pandas as pd

import numpy as np

import random

import itertools

import json

import datetime

import os



# --- ページ設定 ---

st.set_page_config(page_title="シフト作成ツール(3パターン版)", layout="wide")



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



# --- データ管理 ---

def load_settings_from_file():

    if os.path.exists(SETTINGS_FILE):

        try:

            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:

                loaded_data = json.load(f)

            staff_df = pd.DataFrame(loaded_data["staff"])

            config = loaded_data.get("config", {"min_night_staff": 3, "enable_seishain_rule": True, "priority_days": ["土", "日"], "consecutive_penalty_weight": "通常"})

            pairs_df = pd.DataFrame(loaded_data.get("pairs", []))

            if pairs_df.empty: pairs_df = pd.DataFrame(columns=["Staff A", "Staff B", "Type"])

            return staff_df, pd.DataFrame(loaded_data["holidays"]), \

                   datetime.datetime.strptime(loaded_data["date_range"]["start"], "%Y-%m-%d").date(), \

                   datetime.datetime.strptime(loaded_data["date_range"]["end"], "%Y-%m-%d").date(), config, pairs_df

        except: return None, None, None, None, None, None

    return None, None, None, None, None, None



def get_default_date_range():

    today = datetime.date.today()

    start_date = today.replace(day=26)

    if start_date.month == 12: end_date = start_date.replace(year=start_date.year + 1, month=1, day=25)

    else: end_date = start_date.replace(month=start_date.month + 1, day=25)

    return start_date, end_date



# --- 初期化 ---

if 'staff_df' not in st.session_state:

    l_staff, l_holidays, l_start, l_end, l_config, l_pairs = load_settings_from_file()

    if l_staff is not None:

        st.session_state.staff_df, st.session_state.holidays_df = l_staff, l_holidays

        st.session_state.l_start, st.session_state.l_end, st.session_state.config, st.session_state.pairs_df = l_start, l_end, l_config, l_pairs

    else:

        st.session_state.staff_df = pd.DataFrame({"名前": ["西原", "松本"], "レベル": ["リーダー", "スタッフ"], "正社員": [True, True], "朝可": [True, True], "夜可": [True, True], "A": [True, True], "B": [False, True], "C": [False, False], "ネコ": [False, True], "前月末の連勤数": [0, 0], "最大連勤": [4, 4], "公休数": [8, 8]})

        st.session_state.holidays_df = pd.DataFrame(False, index=range(2), columns=[f"Day_{i+1}" for i in range(31)])

        st.session_state.config = {"min_night_staff": 3, "enable_seishain_rule": True, "priority_days": ["土", "日"], "consecutive_penalty_weight": "通常"}

        st.session_state.pairs_df = pd.DataFrame(columns=["Staff A", "Staff B", "Type"])

        st.session_state.l_start, st.session_state.l_end = get_default_date_range()



# --- ロジック ---

def can_cover_required_roles(staff_list, role_map, level_map, min_night_count):

    if sum(1 for s in staff_list if "Night" in role_map[s]) < min_night_count: return False

    if sum(1 for s in staff_list if level_map[s] == "リーダー") < 1: return False

    # ABC要件（簡易版）

    if len(staff_list) < 4: return False

    return True



def assign_roles_smartly(working_indices, role_map):

    assignments = {}

    pool = list(working_indices)

    for s in pool:

        caps = role_map[s]

        if "Neko" in caps: assignments[s] = "ネコ"

        elif "A" in caps: assignments[s] = "A"

        elif "B" in caps: assignments[s] = "B"

        elif "C" in caps: assignments[s] = "C"

        else: assignments[s] = "〇"

    return assignments



def solve_core(staff_df, holidays_df, days_list, config, pairs_df, seed):

    random.seed(seed)

    num_days, num_staff = len(days_list), len(staff_df)

    role_map = {i: {c for c in ["A","B","C","ネコ","Night"] if staff_df.iloc[i].get(c.replace("Night","夜可"))} for i in range(num_staff)}

    level_map = staff_df['レベル'].to_dict()

    name_to_idx = {n: i for i, n in enumerate(staff_df['名前'])}

    

    req_offs = staff_df['公休数'].values

    max_cons = staff_df['最大連勤'].values

    min_night = config.get("min_night_staff", 3)



    # Beam Search

    current_paths = [{'sched': np.zeros((num_staff, num_days)), 'cons': staff_df['前月末の連勤数'].values, 'offs': np.zeros(num_staff), 'score': 0}]

    

    for d_idx, d_obj in enumerate(days_list):

        next_paths = []

        avail = [s for s in range(num_staff) if not holidays_df.iloc[s, d_idx]]

        

        pats = []

        for size in range(4, min(len(avail)+1, 10)):

            pats.extend(list(itertools.combinations(avail, size)))

        random.shuffle(pats)

        pats = pats[:150]



        for path in current_paths:

            for p in pats:

                penalty = 0

                if not can_cover_required_roles(p, role_map, level_map, min_night): penalty += 100000

                

                new_cons = path['cons'].copy()

                new_offs = path['offs'].copy()

                work_mask = np.zeros(num_staff)

                for s in range(num_staff):

                    if s in p:

                        work_mask[s] = 1; new_cons[s] += 1

                        if new_cons[s] > max_cons[s]: penalty += 500000

                    else:

                        new_cons[s] = 0; new_offs[s] += 1

                        # 公休数厳守（超えたら破壊的ペナルティ）

                        if new_offs[s] > req_offs[s]: penalty += 5000000

                

                # 公休数不足も禁止

                days_left = num_days - 1 - d_idx

                for s in range(num_staff):

                    if new_offs[s] + days_left < req_offs[s]: penalty += 5000000



                next_paths.append({'sched': np.hstack([path['sched'], work_mask.reshape(-1,1)]) if d_idx > 0 else work_mask.reshape(-1,1), 

                                   'cons': new_cons, 'offs': new_offs, 'score': path['score'] + penalty})

        

        next_paths.sort(key=lambda x: x['score'])

        current_paths = next_paths[:30]



    best = current_paths[0]

    # 出力整形

    index_names = list(staff_df['名前']) + ["不足"]

    multi_cols = pd.MultiIndex.from_arrays([[str(d.day) for d in days_list] + ["勤(休)"], ["祝" if is_holiday(d) else "月火水木金土日"[d.weekday()] for d in days_list] + [""]])

    res_data = np.full((num_staff+1, num_days+1), "", dtype=object)

    

    for d in range(num_days):

        working = [s for s in range(num_staff) if best['sched'][s, d] == 1]

        roles = assign_roles_smartly(working, role_map)

        for s in range(num_staff):

            if s in working: res_data[s, d] = roles.get(s, "〇")

            else: res_data[s, d] = "／"

        if not can_cover_required_roles(working, role_map, level_map, min_night): res_data[num_staff, d] = "※"

    

    for s in range(num_staff):

        res_data[s, num_days] = f"{int(sum(best['sched'][s, :31]))}({int(best['offs'][s])})"

    

    return pd.DataFrame(res_data, columns=multi_cols, index=index_names), best['score']



# --- UI実装 ---

st.title('📅 シフト作成ツール (3パターン同時生成版)')



with st.sidebar:

    st.header("⚙️ 設定・保存")

    if st.button("💾 設定をサーバーに保存", type="primary"):

        save_dict = {"staff": st.session_state.staff_df.to_dict(), "holidays": st.session_state.holidays_df.to_dict(), "date_range": {"start": st.session_state.l_start.strftime("%Y-%m-%d"), "end": st.session_state.l_end.strftime("%Y-%m-%d")}, "config": st.session_state.config, "pairs": st.session_state.pairs_df.to_dict()}

        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(save_dict, f, ensure_ascii=False, indent=2)

        st.success("保存完了")

    

    start_input = st.date_input("開始日", st.session_state.l_start)

    end_input = st.date_input("終了日", st.session_state.l_end)

    days_list = pd.date_range(start_input, end_input).tolist()



with st.form("settings"):

    with st.expander("🛠 基本設定・ペア設定"):

        c1, c2 = st.columns(2)

        st.session_state.config["min_night_staff"] = c1.number_input("🌙 夜勤最低人数", 1, 10, st.session_state.config["min_night_staff"])

        st.session_state.pairs_df = st.data_editor(st.session_state.pairs_df, num_rows="dynamic", use_container_width=True)

    

    st.markdown("### 1️⃣ スタッフ & 2️⃣ 希望休")

    st.session_state.staff_df = st.data_editor(st.session_state.staff_df, num_rows="dynamic", use_container_width=True)

    # 簡易版のため希望休エディタは省略（実際はstaff_dfと連動して管理）

    if st.form_submit_button("✅ 設定反映"): st.rerun()



st.markdown("### 3️⃣ シフト作成")

if st.button("🚀 3パターンのシフト案を作成する", type="primary"):

    tab1, tab2, tab3 = st.tabs(["案 A (標準)", "案 B (変則)", "案 C (予備)"])

    for i, tab in enumerate([tab1, tab2, tab3]):

        with tab:

            with st.spinner(f"案 {chr(65+i)} を作成中..."):

                res_df, score = solve_core(st.session_state.staff_df, st.session_state.holidays_df, days_list, st.session_state.config, st.session_state.pairs_df, seed=i+100)

                st.dataframe(res_df.style.applymap(lambda v: 'background-color: #ffcccc' if v == '／' else ('background-color: #ff0000; color: white' if v == '※' else '')), use_container_width=True)

                st.download_button(f"📥 案 {chr(65+i)} をダウンロード", res_df.to_csv(encoding="utf-8-sig"), f"shift_plan_{chr(65+i)}.csv")
