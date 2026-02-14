import streamlit as st
import pandas as pd
import numpy as np
import random
import itertools
import json
import datetime
import os

# --- ページ設定 ---
st.set_page_config(page_title="シフト作成ツール(完全版)", layout="wide")

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
            
            # コンフィグの読み込み（欠損キーがあればデフォルト値で補完）
            saved_config = loaded_data.get("config", {})
            default_conf = get_default_config()
            config = {k: saved_config.get(k, v) for k, v in default_conf.items()}
            
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
        st.session_state.config = get_default_config()
        st.session_state.pairs_df = pd.DataFrame(columns=["Staff A", "Staff B", "Type"])
        st.session_state.l_start, st.session_state.l_end = get_default_date_range()

# --- ロジック ---
def can_cover_required_roles(staff_list, role_map, level_map, min_night_count):
    # 夜勤
    if sum(1 for s in staff_list if "Night" in role_map[s]) < min_night_count: return False
    # リーダー
    if sum(1 for s in staff_list if level_map[s] == "リーダー") < 1: return False
    # ABC要件（簡易版）
    if len(staff_list) < 4: return False
    return True

def assign_roles_smartly(working_indices, role_map):
    assignments = {}
    pool = list(working_indices)
    # ネコ、A、B、Cの優先順位で割り当て（簡易ロジック）
    assigned_roles = {"Neko": 0, "A": 0, "B": 0, "C": 0}
    
    # 1. ネコ専任を探す
    for s in pool:
        if "Neko" in role_map[s] and "A" not in role_map[s] and "B" not in role_map[s]:
            assignments[s] = "ネコ"; assigned_roles["Neko"] += 1
    
    # 2. その他割り当て
    for s in pool:
        if s in assignments: continue
        caps = role_map[s]
        if "A" in caps and assigned_roles["A"] == 0: assignments[s] = "A"; assigned_roles["A"] += 1
        elif "B" in caps and assigned_roles["B"] == 0: assignments[s] = "B"; assigned_roles["B"] += 1
        elif "C" in caps and assigned_roles["C"] == 0: assignments[s] = "C"; assigned_roles["C"] += 1
        elif "Neko" in caps and assigned_roles["Neko"] == 0: assignments[s] = "ネコ"; assigned_roles["Neko"] += 1
        elif "A" in caps: assignments[s] = "A"
        elif "B" in caps: assignments[s] = "B"
        elif "C" in caps: assignments[s] = "C"
        else: assignments[s] = "〇"
    return assignments

def solve_core(staff_df, holidays_df, days_list, config, pairs_df, seed):
    random.seed(seed)
    np.random.seed(seed)
    
    num_days, num_staff = len(days_list), len(staff_df)
    role_map = {i: {c for c in ["A","B","C","ネコ","Night"] if staff_df.iloc[i].get(c.replace("Night","夜可"))} for i in range(num_staff)}
    level_map = staff_df['レベル'].to_dict()
    name_to_idx = {n: i for i, n in enumerate(staff_df['名前'])}
    
    req_offs = staff_df['公休数'].values
    max_cons = staff_df['最大連勤'].values
    is_seishain = staff_df['正社員'].values
    
    min_night = config.get("min_night_staff", 3)
    enable_seishain = config.get("enable_seishain_rule", True)
    priority_days_str = config.get("priority_days", [])
    penalty_weight = config.get("consecutive_penalty_weight", "通常")
    cons_penalty_base = 2000 if penalty_weight == "厳格" else (1000 if penalty_weight == "通常" else 500)
    
    weekdays_jp = ["月", "火", "水", "木", "金", "土", "日"]

    # ペア制約
    constraints = []
    if not pairs_df.empty:
        for _, row in pairs_df.iterrows():
            if row["Staff A"] in name_to_idx and row["Staff B"] in name_to_idx:
                constraints.append({"a": name_to_idx[row["Staff A"]], "b": name_to_idx[row["Staff B"]], "type": row["Type"]})

    # Beam Search
    current_paths = [{'sched': np.zeros((num_staff, num_days)), 'cons': staff_df['前月末の連勤数'].values, 
                      'offs': np.zeros(num_staff), 'score': 0}]
    
    for d_idx, d_obj in enumerate(days_list):
        day_str = weekdays_jp[d_obj.weekday()]
        is_weekend = d_obj.weekday() >= 5
        is_priority = day_str in priority_days_str
        
        next_paths = []
        avail = [s for s in range(num_staff) if not holidays_df.iloc[s, d_idx]]
        
        # パターン生成
        pats = []
        for size in range(4, min(len(avail)+1, 10)):
            pats.extend(list(itertools.combinations(avail, size)))
        random.shuffle(pats)
        pats = pats[:150] # 高速化のため制限

        for path in current_paths:
            for p in pats:
                penalty = 0
                
                # 1. 役割要件（夜勤・リーダー） -> 不足しても公休は守るため、ペナルティは公休違反より低く設定
                if not can_cover_required_roles(p, role_map, level_map, min_night):
                    penalty += 50000 
                
                # 2. 優先日
                if is_priority and len(p) <= 4:
                    penalty += 1000

                # 3. ペア制約
                for c in constraints:
                    a_in, b_in = c["a"] in p, c["b"] in p
                    if c["type"] == "NG" and a_in and b_in: penalty += 100000
                    if c["type"] == "Pair" and (a_in != b_in): penalty += 100000

                new_cons = path['cons'].copy()
                new_offs = path['offs'].copy()
                work_mask = np.zeros(num_staff)
                
                for s in range(num_staff):
                    if s in p:
                        work_mask[s] = 1; new_cons[s] += 1
                        # 連勤制約
                        if new_cons[s] > max_cons[s]: 
                             penalty += cons_penalty_base * (new_cons[s] - max_cons[s]) * 10
                    else:
                        new_cons[s] = 0; new_offs[s] += 1
                        # 正社員土日休み抑制
                        if enable_seishain and is_seishain[s] and is_weekend:
                            penalty += 500
                
                # 4. 公休数厳守（ここを絶対的な壁にする）
                days_left = num_days - 1 - d_idx
                for s in range(num_staff):
                    # 休みすぎ
                    if new_offs[s] > req_offs[s]: penalty += 100000000 
                    # 休み不足確定
                    if new_offs[s] + days_left < req_offs[s]: penalty += 100000000

                next_paths.append({'sched': np.hstack([path['sched'], work_mask.reshape(-1,1)]) if d_idx > 0 else work_mask.reshape(-1,1), 
                                   'cons': new_cons, 'offs': new_offs, 'score': path['score'] + penalty})
        
        next_paths.sort(key=lambda x: x['score'])
        current_paths = next_paths[:40]

    best = current_paths[0]
    
    # --- スコアリング（評価機能） ---
    # 100点満点からの減点方式
    eval_score = 100
    insufficient_days = 0
    cons_violations = 0
    weekend_violations = 0
    
    # 出力データ作成
    index_names = list(staff_df['名前']) + ["不足"]
    multi_cols = pd.MultiIndex.from_arrays([[str(d.day) for d in days_list] + ["勤(休)"], ["祝" if is_holiday(d) else weekdays_jp[d.weekday()] for d in days_list] + [""]])
    res_data = np.full((num_staff+1, num_days+1), "", dtype=object)
    
    for d in range(num_days):
        working = [s for s in range(num_staff) if best['sched'][s, d] == 1]
        roles = assign_roles_smartly(working, role_map)
        
        # 不足判定
        if not can_cover_required_roles(working, role_map, level_map, min_night): 
            res_data[num_staff, d] = "※"
            eval_score -= 5 # 不足1日につき-5点
            insufficient_days += 1
        
        for s in range(num_staff):
            if s in working: 
                res_data[s, d] = roles.get(s, "〇")
                # 連勤チェック
                # (簡易的に最終結果から判定は難しいので、ロジック中のペナルティを参照するのが正確だが、ここでは結果から概算)
            else: 
                res_data[s, d] = "／"
                # 正社員土日休みチェック
                day_obj = days_list[d]
                if enable_seishain and is_seishain[s] and (day_obj.weekday() >= 5 or is_holiday(day_obj)):
                    weekend_violations += 1 # カウントのみ（減点は緩く）

    # 評価コメント生成
    comment = []
    if insufficient_days == 0: comment.append("✅ 人員不足なし")
    else: comment.append(f"⚠️ {insufficient_days}日の人員不足あり")
    
    if eval_score < 0: eval_score = 0
    
    # 公休チェック
    holiday_mismatch = 0
    for s in range(num_staff):
        actual_work = int(sum(best['sched'][s, :31]))
        actual_off = int(best['offs'][s])
        res_data[s, num_days] = f"{actual_work}({actual_off})"
        if actual_off != req_offs[s]:
            res_data[s, num_days] += "※"
            holiday_mismatch += 1
            eval_score -= 20 # 公休ズレは大幅減点

    if holiday_mismatch > 0: comment.append(f"⛔ 公休不一致 {holiday_mismatch}名")
    
    evaluation = {
        "score": eval_score,
        "details": f"不足日数: {insufficient_days}日, 公休ズレ: {holiday_mismatch}名",
        "comment": " | ".join(comment)
    }

    return pd.DataFrame(res_data, columns=multi_cols, index=index_names), evaluation

# --- UI実装 ---
st.title('📅 シフト作成ツール (3パターン＋AI採点)')

with st.sidebar:
    st.header("⚙️ 設定・保存")
    if st.button("💾 設定をサーバーに保存", type="primary"):
        save_dict = {
            "staff": st.session_state.staff_df.to_dict(), 
            "holidays": st.session_state.holidays_df.to_dict(), 
            "date_range": {"start": st.session_state.l_start.strftime("%Y-%m-%d"), "end": st.session_state.l_end.strftime("%Y-%m-%d")}, 
            "config": st.session_state.config, 
            "pairs": st.session_state.pairs_df.to_dict()
        }
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(save_dict, f, ensure_ascii=False, indent=2)
        st.success("保存完了")
    
    start_input = st.date_input("開始日", st.session_state.l_start)
    end_input = st.date_input("終了日", st.session_state.l_end)
    days_list = pd.date_range(start_input, end_input).tolist()

with st.form("settings"):
    with st.expander("🛠 基本設定・ペア設定", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.config["min_night_staff"] = st.number_input("🌙 夜勤最低人数", 1, 10, st.session_state.config.get("min_night_staff", 3))
            st.session_state.config["enable_seishain_rule"] = st.checkbox("正社員の土日休み制限", st.session_state.config.get("enable_seishain_rule", True))
            st.session_state.config["consecutive_penalty_weight"] = st.selectbox("連勤ペナルティ", ["通常", "厳格", "緩め"], index=["通常", "厳格", "緩め"].index(st.session_state.config.get("consecutive_penalty_weight", "通常")))
        with c2:
            weekdays = ["月", "火", "水", "木", "金", "土", "日"]
            st.session_state.config["priority_days"] = st.multiselect("優先確保する曜日", weekdays, default=st.session_state.config.get("priority_days", ["土", "日"]))
        
        st.markdown("---")
        st.caption("🤝 ペア設定 (NG / Pair)")
        st.session_state.pairs_df = st.data_editor(st.session_state.pairs_df, num_rows="dynamic", use_container_width=True)
    
    st.markdown("### 1️⃣ スタッフ & 2️⃣ 希望休")
    st.session_state.staff_df = st.data_editor(st.session_state.staff_df, num_rows="dynamic", use_container_width=True)
    if st.form_submit_button("✅ 設定反映"): st.rerun()

st.markdown("### 3️⃣ シフト作成")
if st.button("🚀 3パターンのシフト案を作成する", type="primary"):
    tab1, tab2, tab3 = st.tabs(["案 A", "案 B", "案 C"])
    for i, tab in enumerate([tab1, tab2, tab3]):
        with tab:
            with st.spinner(f"案 {chr(65+i)} を計算中..."):
                res_df, eval_res = solve_core(
                    st.session_state.staff_df, 
                    st.session_state.holidays_df, 
                    days_list, 
                    st.session_state.config, 
                    st.session_state.pairs_df, 
                    seed=i+100
                )
                
                # スコア表示
                c_score, c_info = st.columns([1, 3])
                c_score.metric("AIスコア", f"{eval_res['score']}点")
                c_info.info(f"**評価コメント**: {eval_res['comment']} （{eval_res['details']}）")
                
                st.dataframe(res_df.style.applymap(lambda v: 'background-color: #ffcccc' if v == '／' else ('background-color: #ff0000; color: white' if v == '※' or '※' in str(v) else '')), use_container_width=True)
                st.download_button(f"📥 案 {chr(65+i)} をダウンロード", res_df.to_csv(encoding="utf-8-sig"), f"shift_plan_{chr(65+i)}.csv")
