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

# --- CSSで列幅を強制的に狭くする ---
st.markdown("""
<style>
    div[data-testid="stDataFrame"] div[class^="stDataFrame"] { width: 100%; }
    th { min-width: 30px !important; max-width: 50px !important; padding: 4px !important; font-size: 0.8rem !important; }
    td { min-width: 30px !important; max-width: 50px !important; padding: 4px !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# --- 定数 ---
SETTINGS_FILE = "shift_settings.json"
NIGHT_IDX = 4 
M1_IDX = 5
M2_IDX = 6

# --- データ読み込み・初期化関数 ---
def load_settings_from_file():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            staff_df = pd.DataFrame(loaded_data["staff"])
            # 列補完
            for col in ["朝可", "夜可", "A", "B", "C", "ネコ", "最大連勤"]:
                if col not in staff_df.columns:
                    if col == "朝可": staff_df[col] = True
                    elif col == "夜可": staff_df[col] = False
                    elif col == "最大連勤": staff_df[col] = 4
                    else: staff_df[col] = False

            start_d = None
            end_d = None
            if "date_range" in loaded_data:
                try:
                    start_d = datetime.datetime.strptime(loaded_data["date_range"]["start"], "%Y-%m-%d").date()
                    end_d = datetime.datetime.strptime(loaded_data["date_range"]["end"], "%Y-%m-%d").date()
                except: pass
            
            return staff_df, pd.DataFrame(loaded_data["holidays"]), start_d, end_d
        except Exception:
            return None, None, None, None
    return None, None, None, None

def get_default_date_range():
    today = datetime.date.today()
    if today.day >= 26:
        start_date = today.replace(day=26)
    else:
        start_date = today.replace(day=26)
    if start_date.month == 12:
        end_date = start_date.replace(year=start_date.year + 1, month=1, day=25)
    else:
        end_date = start_date.replace(month=start_date.month + 1, day=25)
    return start_date, end_date

def get_default_data():
    staff_data = {
        "名前": ["正社員A_1", "正社員A_2", "正社員B_1", "正社員B_2", "パート夜", "パート朝1", "パート朝2"],
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

# --- セッション状態の初期化 ---
if 'staff_df' not in st.session_state:
    loaded_staff, loaded_holidays, l_start, l_end = load_settings_from_file()
    if loaded_staff is not None:
        st.session_state.staff_df = loaded_staff
        st.session_state.holidays_df = loaded_holidays
        st.session_state.loaded_start_date = l_start
        st.session_state.loaded_end_date = l_end
        st.toast("📂 設定を読み込みました", icon="✅")
    else:
        d_staff, d_holidays = get_default_data()
        st.session_state.staff_df = d_staff
        st.session_state.holidays_df = d_holidays
        st.session_state.loaded_start_date = None
        st.session_state.loaded_end_date = None

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

def can_cover_required_roles(staff_list, role_map):
    # 条件1: 出勤メンバーの中に「夜可」が3人以上いるか
    night_capable_count = sum(1 for s in staff_list if "Night" in role_map[s])
    if night_capable_count < 3:
        return False

    # 条件2: 出勤メンバーで A, B, C, ネコ が埋まるか
    # 夜勤専従（パート夜）も出勤メンバーに含まれるが、彼らはA,B,C,Nekoを持たない（はず）。
    # 持たない人は割り当て候補にならないだけで、人数にはカウント済み。
    
    # ネコ優先割り当てロジック
    neko_cands = [s for s in staff_list if "Neko" in role_map[s]]
    # パート朝（A,Bを持たないNeko）を優先
    p_neko = [s for s in neko_cands if "A" not in role_map[s] and "B" not in role_map[s]]
    
    neko_fixed = None
    if p_neko: neko_fixed = p_neko[0]
    elif neko_cands: neko_fixed = neko_cands[0]
    
    if neko_fixed is not None:
        rem = [x for x in staff_list if x != neko_fixed]
        if len(rem) < 3: return False
        
        # 簡易チェック: 能力保有者がいるか
        if not any("A" in role_map[x] for x in rem): return False
        if not any("B" in role_map[x] for x in rem): return False
        if not any("C" in role_map[x] for x in rem): return False

        # 厳密チェック（順列）
        # 高速化のため、A,B,C候補を抽出してチェック
        a_cands = [x for x in rem if "A" in role_map[x]]
        b_cands = [x for x in rem if "B" in role_map[x]]
        c_cands = [x for x in rem if "C" in role_map[x]]
        
        # itertoolsで全探索せずとも、A,B,Cがそれぞれ別人で埋まるか確認
        # 3重ループ等は重いので、itertools.permutationsで3人選んでチェック
        # remから3人選ぶ組み合わせ
        for p in itertools.permutations(rem, 3):
             if 'A' in role_map[p[0]] and 'B' in role_map[p[1]] and 'C' in role_map[p[2]]:
                 return True
        return False
    else:
        # ネコ役も誰か（正社員等）がやる場合
        if len(staff_list) < 4: return False
        for p in itertools.permutations(staff_list, 4):
             if 'Neko' in role_map[p[0]] and 'A' in role_map[p[1]] and 'B' in role_map[p[2]] and 'C' in role_map[p[3]]:
                 return True
        return False

def get_possible_day_patterns(available_staff):
    patterns = []
    # 最低4人（A,B,C,Neko兼務）から
    for size in range(4, min(len(available_staff)+1, 10)):
        for subset in itertools.combinations(available_staff, size):
            patterns.append(subset)
    return patterns

def assign_roles_smartly(working_indices, role_map):
    assignments = {}
    
    # 割り当て優先順位:
    # 1. ネコ, A, B, C を埋める（必須）
    # 2. 余った人で、かつ夜勤専従（日勤役割なし）がいれば '〇'
    # 3. それ以外は能力に応じて表示
    
    pool = list(working_indices)
    
    neko_cands = [s for s in pool if "Neko" in role_map[s]]
    p_neko = [s for s in neko_cands if "A" not in role_map[s] and "B" not in role_map[s]]
    
    neko_fixed = None
    if p_neko: neko_fixed = p_neko[0]
    elif neko_cands: neko_fixed = neko_cands[0]
    
    found_strict = False
    
    if neko_fixed is not None:
        rem = [x for x in pool if x != neko_fixed]
        for p in itertools.permutations(rem, 3):
            if 'A' in role_map[p[0]] and 'B' in role_map[p[1]] and 'C' in role_map[p[2]]:
                assignments[neko_fixed] = 'ネコ'
                assignments[p[0]] = 'A'
                assignments[p[1]] = 'B'
                assignments[p[2]] = 'C'
                found_strict = True
                
                # 余り人員の処理
                unassigned = [x for x in rem if x not in p]
                for ex in unassigned:
                    # 夜勤専従判定（日勤役割がない かつ Nightあり）
                    # または、ユーザー要望「夜だけの人は役割は〇」
                    has_day_role = any(r in role_map[ex] for r in ["A","B","C","Neko"])
                    if not has_day_role and "Night" in role_map[ex]:
                        assignments[ex] = '〇'
                    else:
                        # 日勤能力があるならそれを表示（C優先）
                        caps = role_map[ex]
                        if 'C' in caps: assignments[ex] = 'C'
                        elif 'B' in caps: assignments[ex] = 'B'
                        elif 'A' in caps: assignments[ex] = 'A'
                        elif 'Neko' in caps: assignments[ex] = 'ネコ'
                        elif "Night" in role_map[ex]: assignments[ex] = '〇' # 最終手段
                break
    else:
        for p in itertools.permutations(pool, 4):
            if 'Neko' in role_map[p[0]] and 'A' in role_map[p[1]] and 'B' in role_map[p[2]] and 'C' in role_map[p[3]]:
                assignments[p[0]] = 'ネコ'; assignments[p[1]] = 'A'; assignments[p[2]] = 'B'; assignments[p[3]] = 'C'
                found_strict = True
                unassigned = [x for x in pool if x not in p]
                for ex in unassigned:
                    has_day_role = any(r in role_map[ex] for r in ["A","B","C","Neko"])
                    if not has_day_role and "Night" in role_map[ex]:
                        assignments[ex] = '〇'
                    else:
                        caps = role_map[ex]
                        if 'C' in caps: assignments[ex] = 'C'
                        elif 'B' in caps: assignments[ex] = 'B'
                        elif 'A' in caps: assignments[ex] = 'A'
                break
    
    if not found_strict:
        # ベストエフォート（不足時の表示）
        unassigned = set(pool)
        for s in pool:
            if s in unassigned and 'A' in role_map[s]: assignments[s] = 'A'; unassigned.remove(s); break
        for s in pool:
            if s in unassigned and 'B' in role_map[s]: assignments[s] = 'B'; unassigned.remove(s); break
        if neko_fixed and neko_fixed in unassigned:
            assignments[neko_fixed] = 'ネコ'; unassigned.remove(neko_fixed)
        else:
            for s in list(unassigned):
                if 'Neko' in role_map[s]: assignments[s] = 'ネコ'; unassigned.remove(s); break
        for s in list(unassigned):
            if "Night" in role_map[s] and not any(r in role_map[s] for r in ["A","B","C","Neko"]):
                 assignments[s] = '〇'; unassigned.discard(s)
            elif 'C' in role_map[s]: assignments[s] = 'C'; unassigned.discard(s)

    return assignments

def solve_schedule_from_ui(staff_df, holidays_df, days_list):
    staff_df = staff_df.dropna(subset=['名前'])
    staff_df = staff_df[staff_df['名前'] != '']
    staff_df = staff_df.reset_index(drop=True)
    num_days = len(days_list)
    num_staff = len(staff_df)
    if num_staff == 0: return None
    
    role_map = get_role_map_from_df(staff_df)

    try:
        col_prev_cons = "前月末の連勤数" if "前月末の連勤数" in staff_df.columns else "先月からの連勤"
        initial_cons = pd.to_numeric(staff_df[col_prev_cons], errors='coerce').fillna(0).astype(int).values
        req_offs = pd.to_numeric(staff_df['公休数'], errors='coerce').fillna(0).astype(int).values
        max_cons_limits = pd.to_numeric(staff_df['最大連勤'], errors='coerce').fillna(4).astype(int).values
    except: return None 
    
    fixed_shifts = np.full((num_staff, num_days), '', dtype=object)
    holidays_df = holidays_df.reset_index(drop=True)
    
    for d_idx in range(num_days):
        col_name = f"Day_{d_idx+1}"
        if col_name in holidays_df.columns:
            col_data = holidays_df[col_name].values
            for s_idx in range(num_staff):
                if s_idx < len(col_data): 
                    if col_data[s_idx] == True or col_data[s_idx] == '×': fixed_shifts[s_idx, d_idx] = '×'
    
    day_patterns = []
    for d in range(num_days):
        avail = [s for s in range(num_staff) if fixed_shifts[s, d] != '×']
        pats = get_possible_day_patterns(avail)
        random.shuffle(pats)
        day_patterns.append(pats)

    current_paths = [{'sched': np.zeros((num_staff, num_days), dtype=int), 'cons': initial_cons.copy(), 'offs': np.zeros(num_staff, dtype=int), 'off_cons': np.zeros(num_staff, dtype=int), 'score': 0}]
    BEAM_WIDTH = 200
    
    for d in range(num_days):
        next_paths = []
        patterns = day_patterns[d]
        
        valid_pats = [p for p in patterns if can_cover_required_roles(p, role_map)]
        invalid_pats = [p for p in patterns if not can_cover_required_roles(p, role_map)]
        use_patterns = valid_pats[:200] + invalid_pats[:50]
        
        for path in current_paths:
            for pat in use_patterns:
                new_cons = path['cons'].copy(); new_offs = path['offs'].copy(); new_off_cons = path['off_cons'].copy(); penalty = 0; violation = False
                
                # 不足ペナルティ
                if not can_cover_required_roles(pat, role_map): penalty += 50000
                
                work_mask = np.zeros(num_staff, dtype=int)
                for s in pat: work_mask[s] = 1
                
                for s in range(num_staff):
                    limit = max_cons_limits[s]
                    if work_mask[s] == 1:
                        new_cons[s] += 1; new_off_cons[s] = 0
                        if new_cons[s] > limit:
                            if new_cons[s] == limit + 1: penalty += 1000
                            else: violation = True; break
                        elif new_cons[s] == limit: penalty += 50
                    else:
                        new_cons[s] = 0; new_offs[s] += 1; new_off_cons[s] += 1
                        if new_off_cons[s] >= 3:
                            penalty += 100
                            # パート朝判定（役割で判定）
                            if "Neko" in role_map[s] and "C" in role_map[s] and "A" not in role_map[s]: penalty += 200
                
                if violation: continue
                days_left = num_days - 1 - d
                if np.any(new_offs > req_offs): violation = True
                if np.any(new_offs + days_left < req_offs): violation = True
                if violation: continue
                expected = req_offs * ((d+1)/num_days)
                penalty += np.sum(np.abs(new_offs - expected)) * 10
                new_sched = path['sched'].copy(); new_sched[:, d] = work_mask
                next_paths.append({'sched': new_sched, 'cons': new_cons, 'offs': new_offs, 'off_cons': new_off_cons, 'score': path['score'] + penalty})
        
        next_paths.sort(key=lambda x: x['score'])
        if not next_paths: return None
        current_paths = next_paths[:BEAM_WIDTH]
    
    best_path = current_paths[0]; final_sched = best_path['sched']
    weekdays_jp = ["(月)", "(火)", "(水)", "(木)", "(金)", "(土)", "(日)"]
    output_cols = [f"{d.month}/{d.day}{weekdays_jp[d.weekday()]}" for d in days_list]
    output_data = np.full((num_staff + 1, num_days), "", dtype=object)
    insufficient_row_idx = num_staff
    for d in range(num_days):
        working = [s for s in range(num_staff) if final_sched[s, d] == 1]
        roles = assign_roles_smartly(working, role_map)
        is_insufficient = False
        if not can_cover_required_roles(working, role_map): is_insufficient = True
        for s in range(num_staff):
            if s in working:
                if s in roles: output_data[s, d] = roles[s]
                else:
                    caps = role_map[s]
                    if 'C' in caps: output_data[s, d] = 'C'
                    elif 'B' in caps: output_data[s, d] = 'B'
                    elif 'A' in caps: output_data[s, d] = 'A'
                    else: output_data[s, d] = 'C'
            else: output_data[s, d] = '×' if fixed_shifts[s, d] == '×' else '／'
        if is_insufficient: output_data[insufficient_row_idx, d] = "※"
    index_names = list(staff_df['名前']) + ["不足"]
    return pd.DataFrame(output_data, columns=output_cols, index=index_names)

def highlight_cells(val):
    if val == '／': return 'background-color: #ffcccc; color: black'
    elif val == '×': return 'background-color: #d9d9d9; color: gray'
    elif val == '※': return 'background-color: #ff0000; color: white; font-weight: bold'
    elif val == 'A': return 'background-color: #ccffff; color: black'
    elif val == 'B': return 'background-color: #ccffcc; color: black'
    elif val == 'C': return 'background-color: #ffffcc; color: black'
    elif val == 'ネコ': return 'background-color: #ffe5cc; color: black'
    elif val == '〇': return 'background-color: #e6e6fa; color: black'
    return ''

# ==========================================
# UI実装
# ==========================================
st.title('📅 シフト作成ツール')

# CSS設定（列幅縮小）
st.markdown("""
<style>
    div[data-testid="stDataFrame"] div[class^="stDataFrame"] { width: 100%; }
    th { min-width: 30px !important; max-width: 50px !important; padding: 4px !important; font-size: 0.8rem !important; }
    td { min-width: 30px !important; max-width: 50px !important; padding: 4px !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

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
        save_data = {
            "staff": clean_staff_df.to_dict(),
            "holidays": st.session_state.holidays_df.to_dict(),
            "date_range": {
                "start": start_input.strftime("%Y-%m-%d"),
                "end": end_input.strftime("%Y-%m-%d")
            }
        }
        try:
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            st.success("保存しました！")
        except Exception as e:
            st.error(f"保存失敗: {e}")

    st.markdown("---")
    st.subheader("📥 バックアップ")
    clean_staff_df = st.session_state.staff_df.dropna(subset=['名前'])
    clean_staff_df = clean_staff_df[clean_staff_df['名前'] != '']
    current_data = {
        "staff": clean_staff_df.to_dict(),
        "holidays": st.session_state.holidays_df.to_dict(),
        "date_range": {
            "start": start_input.strftime("%Y-%m-%d"),
            "end": end_input.strftime("%Y-%m-%d")
        }
    }
    json_str = json.dumps(current_data, ensure_ascii=False)
    st.download_button("設定ファイルDL", json_str, "shift_settings.json", "application/json")
    
    uploaded_json = st.file_uploader("設定ファイル読込", type=["json"])
    if uploaded_json is not None:
        try:
            loaded_data = json.load(uploaded_json)
            df_new = pd.DataFrame(loaded_data["staff"])
            for col in ["朝可", "夜可", "A", "B", "C", "ネコ", "最大連勤"]:
                if col not in df_new.columns:
                    if col == "最大連勤": df_new[col] = 4
                    else: df_new[col] = False
            if "先月からの連勤" in df_new.columns:
                df_new["前月末の連勤数"] = df_new["先月からの連勤"]
            
            st.session_state.staff_df = df_new
            st.session_state.holidays_df = pd.DataFrame(loaded_data["holidays"])
            if "date_range" in loaded_data:
                st.session_state.loaded_start_date = datetime.datetime.strptime(loaded_data["date_range"]["start"], "%Y-%m-%d").date()
                st.session_state.loaded_end_date = datetime.datetime.strptime(loaded_data["date_range"]["end"], "%Y-%m-%d").date()
                st.rerun()
            st.success("読み込み完了")
        except:
            st.error("読込エラー")

# --- メインエリア ---
st.markdown("### 1️⃣ スタッフ設定")

edited_staff_df = st.data_editor(
    st.session_state.staff_df,
    num_rows="dynamic",
    use_container_width=True,
    height=300,
    column_config={
        "朝可": st.column_config.CheckboxColumn("朝", width="small", default=True),
        "夜可": st.column_config.CheckboxColumn("夜", width="small", default=False),
        "A": st.column_config.CheckboxColumn("A", width="small", default=False),
        "B": st.column_config.CheckboxColumn("B", width="small", default=False),
        "C": st.column_config.CheckboxColumn("C", width="small", default=False),
        "ネコ": st.column_config.CheckboxColumn("🐱", width="small", default=False),
        "前月末の連勤数": st.column_config.NumberColumn("前月連勤", width="small"),
        "最大連勤": st.column_config.NumberColumn("最大連勤", width="small", default=4),
        "公休数": st.column_config.NumberColumn("公休", width="small"),
        "名前": st.column_config.TextColumn("名前", width="medium"),
    }
)
st.session_state.staff_df = edited_staff_df

valid_staff_count = len(edited_staff_df[edited_staff_df['名前'].notna() & (edited_staff_df['名前'] != "")])
current_holiday_rows = len(st.session_state.holidays_df)
if valid_staff_count > current_holiday_rows:
    rows_to_add = valid_staff_count - current_holiday_rows
    new_data = pd.DataFrame(False, index=range(rows_to_add), columns=st.session_state.holidays_df.columns)
    st.session_state.holidays_df = pd.concat([st.session_state.holidays_df, new_data], ignore_index=True)
elif valid_staff_count < current_holiday_rows:
    st.session_state.holidays_df = st.session_state.holidays_df.iloc[:valid_staff_count]

st.markdown("### 2️⃣ 希望休入力")
holiday_cols = [f"Day_{i+1}" for i in range(num_days)]
display_holidays_df = st.session_state.holidays_df.reindex(columns=holiday_cols, fill_value=False)
valid_names = edited_staff_df[edited_staff_df['名前'].notna() & (edited_staff_df['名前'] != "")]['名前']
if len(valid_names) == len(display_holidays_df): display_holidays_df.index = valid_names

edited_holidays_grid = st.data_editor(
    display_holidays_df,
    use_container_width=True,
    column_config={
        col: st.column_config.CheckboxColumn(
            f"{days_list[i].day}({['月','火','水','木','金','土','日'][days_list[i].weekday()]})", 
            default=False,
            width="small"
        ) for i, col in enumerate(holiday_cols)
    }
)
st.session_state.holidays_df = edited_holidays_grid.reset_index(drop=True)

st.markdown("### 3️⃣ シフト作成")
if st.button("シフトを作成する", type="primary"):
    with st.spinner("AIがシフトパズルを解いています...🧩"):
        try:
            result_df = solve_schedule_from_ui(edited_staff_df, edited_holidays_grid, days_list)
            if result_df is not None:
                st.success("作成完了！")
                styled_df = result_df.fillna("").style.map(highlight_cells)
                st.dataframe(styled_df, use_container_width=True, height=600)
                csv = result_df.to_csv().encode('utf-8-sig')
                st.download_button("CSVダウンロード", csv, "shift_result.csv", "text/csv")
            else:
                st.error("条件を満たすシフトが見つかりませんでした。最大連勤数などを緩和してください。")
        except Exception as e:
            st.error(f"エラー: {e}")
