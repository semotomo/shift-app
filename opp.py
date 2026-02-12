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
    th, td { padding: 2px 4px !important; font-size: 13px !important; text-align: center !important; }
    div[data-testid="stDataFrame"] th { white-space: pre-wrap !important; vertical-align: bottom !important; line-height: 1.3 !important; }
    div[data-testid="stDataFrame"] th span { white-space: pre-wrap !important; display: inline-block !important; }
    th[aria-label="名前"], td[aria-label="名前"] { max-width: 100px !important; min-width: 100px !important; }
    th[aria-label="社員"], td[aria-label="社員"],
    th[aria-label="朝"], td[aria-label="朝"],
    th[aria-label="夜"], td[aria-label="夜"],
    th[aria-label="A"], td[aria-label="A"],
    th[aria-label="B"], td[aria-label="B"],
    th[aria-label="C"], td[aria-label="C"],
    th[aria-label="🐱"], td[aria-label="🐱"] { max-width: 25px !important; min-width: 25px !important; }
</style>
""", unsafe_allow_html=True)

# --- 定数 ---
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

def get_default_config():
    return {
        "min_night_staff": 3,
        "min_a_staff": 1,
        "enable_seishain_rule": True,
        "enable_interval_rule": True, # 夜勤後の朝勤禁止
        "priority_days": ["土", "日"],
        "consecutive_penalty_weight": "通常"
    }

def load_settings_from_file():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            staff_df = pd.DataFrame(loaded_data["staff"])
            for col in ["正社員", "朝可", "夜可", "A", "B", "C", "ネコ", "最大連勤"]:
                if col not in staff_df.columns:
                    staff_df[col] = 4 if col == "最大連勤" else (True if col == "朝可" else False)
            start_d = datetime.datetime.strptime(loaded_data["date_range"]["start"], "%Y-%m-%d").date() if "date_range" in loaded_data else None
            end_d = datetime.datetime.strptime(loaded_data["date_range"]["end"], "%Y-%m-%d").date() if "date_range" in loaded_data else None
            return staff_df, pd.DataFrame(loaded_data["holidays"]), start_d, end_d, loaded_data.get("config", get_default_config())
        except: return None, None, None, None, None
    return None, None, None, None, None

def get_default_date_range():
    today = datetime.date.today()
    start_date = today.replace(day=26)
    end_date = start_date.replace(month=start_date.month + 1, day=25) if start_date.month < 12 else start_date.replace(year=start_date.year + 1, month=1, day=25)
    return start_date, end_date

# --- セッション初期化 ---
if 'staff_df' not in st.session_state:
    l_staff, l_holidays, l_start, l_end, l_config = load_settings_from_file()
    if l_staff is not None:
        st.session_state.staff_df, st.session_state.holidays_df, st.session_state.loaded_start_date, st.session_state.loaded_end_date, st.session_state.config = l_staff, l_holidays, l_start, l_end, l_config
    else:
        st.session_state.staff_df = pd.DataFrame({"名前": ["西原", "松本", "中島", "山下", "下尾", "原", "松尾"], "正社員": [True]*4+[False]*3, "朝可": [True]*7, "夜可": [True]*5+[False]*2, "A": [True,True,False,False,False,False,False], "B": [False,True,True,True,False,False,False], "C": [False,False,True,True,False,True,True], "ネコ": [False,True,True,True,False,True,True], "前月末の連勤数": [0,0,0,0,0,0,0], "最大連勤": [4,4,4,4,3,4,3], "公休数": [8,8,8,8,13,9,15]})
        st.session_state.holidays_df = pd.DataFrame(False, index=range(7), columns=[f"Day_{i+1}" for i in range(31)])
        st.session_state.config = get_default_config()

# --- 判定ロジック ---
def can_cover_required_roles(staff_list, role_map, min_night, min_a=1):
    if sum(1 for s in staff_list if "Night" in role_map[s]) < min_night: return False
    if sum(1 for s in staff_list if "A" in role_map[s]) < min_a: return False
    neko_cands = [s for s in staff_list if "Neko" in role_map[s]]
    if not neko_cands: return False
    rem = [x for x in staff_list if x != neko_cands[0]]
    if not all(any(r in role_map[x] for x in rem) for r in ["A", "B", "C"]): return False
    return True

def assign_roles_smartly(working_indices, role_map):
    assignments = {}
    pool = list(working_indices)
    for r in ['ネコ', 'A', 'B', 'C']:
        target = 'Neko' if r == 'ネコ' else r
        for s in pool:
            if target in role_map[s] and s not in assignments:
                assignments[s] = r; break
    for s in pool:
        if s not in assignments: assignments[s] = '〇' if "Night" in role_map[s] else 'C'
    return assignments

# --- シフト作成エンジン ---
def solve_schedule(staff_df, holidays_df, days_list, config):
    staff_df = staff_df.dropna(subset=['名前']).reset_index(drop=True)
    num_days, num_staff = len(days_list), len(staff_df)
    role_map = {i: {r for r in ['A','B','C','Neko','Night'] if staff_df.iloc[i][r if r!='Neko' else 'ネコ']} for i in range(num_staff)}
    
    # 以前の夜勤フラグ管理用
    was_night = np.zeros(num_staff, dtype=bool)
    
    current_paths = [{'sched': np.zeros((num_staff, num_days), dtype=int), 'cons': staff_df['前月末の連勤数'].values, 'offs': np.zeros(num_staff), 'score': 0, 'was_night': was_night}]
    
    for d in range(num_days):
        is_weekend = days_list[d].weekday() >= 5
        day_name = ["月","火","水","木","金","土","日"][days_list[d].weekday()]
        next_paths = []
        avail = [s for s in range(num_staff) if not holidays_df.iloc[s, d]]
        
        pats = [p for size in range(4, min(len(avail)+1, 8)) for p in itertools.combinations(avail, size)]
        random.shuffle(pats)
        
        for path in current_paths[:150]:
            for pat in pats[:40]:
                penalty = 0
                if not can_cover_required_roles(pat, role_map, config['min_night_staff'], config['min_a_staff']): penalty += 800000
                
                work_mask = np.zeros(num_staff, dtype=int)
                for s in pat: work_mask[s] = 1
                
                new_cons, new_offs, new_was_night = path['cons'].copy(), path['offs'].copy(), np.zeros(num_staff, dtype=bool)
                for s in range(num_staff):
                    if work_mask[s]:
                        # インターバル規制: 前日夜勤(Night役)かつ当日出勤で、当日朝しかできない場合は重いペナルティ
                        if config['enable_interval_rule'] and path['was_night'][s]:
                             if not staff_df.iloc[s]['夜可']: penalty += 500000
                        
                        new_cons[s] += 1
                        if new_cons[s] > staff_df.iloc[s]['最大連勤']: penalty += 500000
                        # 当日の役割がNightか判定（簡易的にNight持ちならフラグを立てる）
                        if "Night" in role_map[s]: new_was_night[s] = True
                    else:
                        new_cons[s] = 0; new_offs[s] += 1
                        if config['enable_seishain_rule'] and is_seishain[s] and is_weekend: penalty += 500
                
                for s in range(num_staff):
                    if new_offs[s] > staff_df.iloc[s]['公休数']: penalty += 1000000
                    if new_offs[s] + (num_days - 1 - d) < staff_df.iloc[s]['公休数']: penalty += 1000000

                next_paths.append({'sched': np.hstack([path['sched'], work_mask.reshape(-1,1)]), 'cons': new_cons, 'offs': new_offs, 'score': path['score']+penalty, 'was_night': new_was_night})
        
        next_paths.sort(key=lambda x: x['score'])
        current_paths = next_paths[:150]
        if not current_paths: return None, 9999999
        
    best = current_paths[0]
    output = np.full((num_staff+1, num_days+1), "", dtype=object)
    for d in range(num_days):
        working = [s for s in range(num_staff) if best['sched'][s, d]]
        roles = assign_roles_smartly(working, role_map)
        for s in range(num_staff):
            output[s, d] = roles.get(s, '／' if not holidays_df.iloc[s, d] else '×')
        if not can_cover_required_roles(working, role_map, config['min_night_staff']): output[num_staff, d] = "※"
    
    for s in range(num_staff):
        off = sum(1 for x in output[s, :num_days] if x in ['／', '×'])
        output[s, num_days] = f"{num_days-off}({off})" + ("※" if off != staff_df.iloc[s]['公休数'] else "")
        
    return pd.DataFrame(output, index=list(staff_df['名前'])+["不足"], columns=pd.MultiIndex.from_arrays([[str(d.day) for d in days_list]+["勤(休)"], ["祝" if is_holiday(d) else ["月","火","水","木","金","土","日"][d.weekday()] for d in days_list]+[""]])), best['score']

# --- UI構築 ---
st.title('📅 シフト作成ツール')

with st.sidebar:
    if st.button("💾 設定をサーバーに保存", type="primary"):
        save_data = {"staff": st.session_state.staff_df.to_dict(), "holidays": st.session_state.holidays_df.to_dict(), "date_range": {"start": st.date_input("開始日", st.session_state.get('loaded_start_date', datetime.date.today())).strftime("%Y-%m-%d"), "end": st.date_input("終了日", st.session_state.get('loaded_end_date', datetime.date.today())).strftime("%Y-%m-%d")}, "config": st.session_state.config}
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(save_data, f, ensure_ascii=False, indent=2)
        st.success("保存完了")

    st.header("📅 日付設定")
    d_start, d_end = get_default_date_range()
    start_input = st.date_input("開始日", st.session_state.get('loaded_start_date', d_start))
    end_input = st.date_input("終了日", st.session_state.get('loaded_end_date', d_end))
    days_list = pd.date_range(start_input, end_input).tolist()

with st.form("main_form"):
    with st.expander("🛠 基本設定（クリックで開閉）"):
        c1, c2 = st.columns(2)
        st.session_state.config["min_night_staff"] = c1.number_input("🌙 夜勤の最低人数", 1, 10, st.session_state.config["min_night_staff"])
        st.session_state.config["min_a_staff"] = c1.number_input("🅰️ 役割Aの最低人数", 0, 10, st.session_state.config.get("min_a_staff", 1))
        st.session_state.config["enable_interval_rule"] = c2.checkbox("🛌 夜勤の翌日の朝勤を禁止する", st.session_state.config["enable_interval_rule"])
        st.session_state.config["enable_seishain_rule"] = c2.checkbox("👔 正社員の土日休み制限", st.session_state.config["enable_seishain_rule"])

    st.markdown("### 1️⃣ スタッフ設定")
    edited_staff = st.data_editor(st.session_state.staff_df, num_rows="dynamic", use_container_width=True, hide_index=True)
    
    st.markdown("### 2️⃣ 希望休入力")
    edited_holidays = st.data_editor(st.session_state.holidays_df.iloc[:len(edited_staff)], use_container_width=True, hide_index=True)
    
    if st.form_submit_button("✅ 設定反映"):
        st.session_state.staff_df, st.session_state.holidays_df = edited_staff, edited_holidays
        st.rerun()

if st.button("シフトを作成する"):
    with st.spinner("計算中..."):
        res_df, score = solve_schedule(st.session_state.staff_df, st.session_state.holidays_df, days_list, st.session_state.config)
        if res_df is not None:
            if score >= 500000: st.warning("⚠️ 一部ルールを緩和して作成しました。")
            else: st.success("✨ 条件クリア！")
            st.dataframe(res_df.style.apply(lambda x: ["background-color: #ffcccc" if v=='／' else "background-color: #e6f7ff" if "土" in str(x.name) else "" for v in x], axis=0), use_container_width=True)
