import streamlit as st
import pandas as pd
import numpy as np
import random
import itertools
import json
import datetime

# --- ページ設定 ---
st.set_page_config(page_title="シフト作成ツール(入力版)", layout="wide")

# --- セッション状態の初期化 ---
if 'staff_df' not in st.session_state:
    default_data = {
        "名前": ["正社員A_1", "正社員A_2", "正社員B_1", "正社員B_2", "パート夜", "パート朝1", "パート朝2"],
        "役割(カンマ区切り)": ["A", "A,B,Neko", "B,C,Neko", "B,C,Neko", "Night", "Neko,C", "Neko,C"],
        "先月からの連勤": [0, 5, 1, 0, 0, 2, 2],
        "公休数": [8, 8, 8, 8, 13, 9, 15]
    }
    st.session_state.staff_df = pd.DataFrame(default_data)

if 'holidays_df' not in st.session_state:
    num_staff = len(st.session_state.staff_df)
    st.session_state.holidays_df = pd.DataFrame(False, index=range(num_staff), columns=[f"Day_{i+1}" for i in range(31)])

# --- 定数 ---
FULL_TIME_IDXS = [0, 1, 2, 3] 
NIGHT_IDX = 4
M1_IDX = 5
M2_IDX = 6

# --- ロジック関数 ---

def parse_roles(role_str):
    if not isinstance(role_str, str): return set()
    return {r.strip() for r in role_str.split(',')}

def can_cover_required_roles(staff_list, role_map):
    if NIGHT_IDX in staff_list:
        if sum(1 for s in staff_list if s in FULL_TIME_IDXS) < 2: return False
    
    pool = [s for s in staff_list if s != NIGHT_IDX]
    
    neko_fixed = None
    if M1_IDX in pool: neko_fixed = M1_IDX
    elif M2_IDX in pool: neko_fixed = M2_IDX
    
    if neko_fixed is not None:
        rem = [x for x in pool if x != neko_fixed]
        if len(rem) < 3: return False
        for p in itertools.permutations(rem, 3):
            if 'A' in role_map[p[0]] and 'B' in role_map[p[1]] and 'C' in role_map[p[2]]:
                return True
    else:
        if len(pool) < 4: return False
        for p in itertools.permutations(pool, 4):
            if 'Neko' in role_map[p[0]] and 'A' in role_map[p[1]] and 'B' in role_map[p[2]] and 'C' in role_map[p[3]]:
                return True
    return False

def get_possible_day_patterns(available_staff):
    patterns = []
    for size in range(3, 8):
        for subset in itertools.combinations(available_staff, size):
            patterns.append(subset)
    return patterns

def assign_roles_smartly(working_indices, role_map):
    assignments = {}
    if NIGHT_IDX in working_indices: assignments[NIGHT_IDX] = '〇'
    
    pool = [s for s in working_indices if s != NIGHT_IDX]
    if not pool: return assignments
    
    neko_fixed = None
    if M1_IDX in pool: neko_fixed = M1_IDX
    elif M2_IDX in pool: neko_fixed = M2_IDX
    
    found_strict = False
    
    if neko_fixed is not None:
        rem = [x for x in pool if x != neko_fixed]
        for p in itertools.permutations(rem, 3):
            if 'A' in role_map[p[0]] and 'B' in role_map[p[1]] and 'C' in role_map[p[2]]:
                assignments[neko_fixed] = 'ネコ'
                assignments[p[0]] = 'A'; assignments[p[1]] = 'B'; assignments[p[2]] = 'C'
                found_strict = True
                for ex in rem:
                    if ex not in p:
                        caps = role_map[ex]
                        if 'C' in caps: assignments[ex] = 'C'
                        elif 'B' in caps: assignments[ex] = 'B'
                        elif 'A' in caps: assignments[ex] = 'A'
                break
    else:
        for p in itertools.permutations(pool, 4):
            if 'Neko' in role_map[p[0]] and 'A' in role_map[p[1]] and 'B' in role_map[p[2]] and 'C' in role_map[p[3]]:
                assignments[p[0]] = 'ネコ'; assignments[p[1]] = 'A'; assignments[p[2]] = 'B'; assignments[p[3]] = 'C'
                found_strict = True
                for ex in pool:
                    if ex not in p:
                        caps = role_map[ex]
                        if 'C' in caps: assignments[ex] = 'C'
                        elif 'B' in caps: assignments[ex] = 'B'
                        elif 'A' in caps: assignments[ex] = 'A'
                break

    if found_strict: return assignments

    unassigned = set(pool)
    for s in pool:
        if s in unassigned and 'A' in role_map[s]:
            assignments[s] = 'A'; unassigned.remove(s); break
    for s in pool:
        if s in unassigned and 'B' in role_map[s]:
            assignments[s] = 'B'; unassigned.remove(s); break
    
    if M1_IDX in unassigned: assignments[M1_IDX] = 'ネコ'; unassigned.remove(M1_IDX)
    elif M2_IDX in unassigned: assignments[M2_IDX] = 'ネコ'; unassigned.remove(M2_IDX)
    else:
        for s in pool:
            if s in unassigned and 'Neko' in role_map[s]:
                assignments[s] = 'ネコ'; unassigned.remove(s); break
                
    for s in list(unassigned):
        caps = role_map[s]
        if 'C' in caps: assignments[s] = 'C'
        elif 'B' in caps: assignments[s] = 'B'
        elif 'A' in caps: assignments[s] = 'A'
        elif 'Neko' in caps: assignments[s] = 'ネコ'
        
    return assignments

def solve_schedule_from_ui(staff_df, holidays_df, days_list):
    # クリーンアップ
    staff_df = staff_df.dropna(subset=['名前'])
    staff_df = staff_df[staff_df['名前'] != '']
    
    num_days = len(days_list)
    num_staff = len(staff_df)
    
    if num_staff == 0: return None
    
    role_map = {}
    staff_df = staff_df.reset_index(drop=True)
    
    for i, row in staff_df.iterrows():
        role_map[i] = parse_roles(str(row['役割(カンマ区切り)']))

    try:
        initial_cons = pd.to_numeric(staff_df['先月からの連勤'], errors='coerce').fillna(0).astype(int).values
        req_offs = pd.to_numeric(staff_df['公休数'], errors='coerce').fillna(0).astype(int).values
    except:
        return None 
    
    fixed_shifts = np.full((num_staff, num_days), '', dtype=object)
    holidays_df = holidays_df.reset_index(drop=True)
    
    for d_idx in range(num_days):
        col_name = f"Day_{d_idx+1}"
        if col_name in holidays_df.columns:
            col_data = holidays_df[col_name].values
            for s_idx in range(num_staff):
                if s_idx < len(col_data): 
                    if col_data[s_idx] == True or col_data[s_idx] == '×':
                        fixed_shifts[s_idx, d_idx] = '×'
    
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
        'score': 0
    }]
    BEAM_WIDTH = 150
    
    for d in range(num_days):
        next_paths = []
        patterns = day_patterns[d]
        
        valid_pats = [p for p in patterns if can_cover_required_roles(p, role_map)]
        invalid_pats = [p for p in patterns if not can_cover_required_roles(p, role_map)]
        use_patterns = valid_pats[:150] + invalid_pats[:30]
        
        for path in current_paths:
            for pat in use_patterns:
                new_cons = path['cons'].copy()
                new_offs = path['offs'].copy()
                new_off_cons = path['off_cons'].copy()
                penalty = 0
                violation = False
                
                if not can_cover_required_roles(pat, role_map):
                    penalty += 50000
                
                work_mask = np.zeros(num_staff, dtype=int)
                for s in pat: work_mask[s] = 1
                
                for s in range(num_staff):
                    if work_mask[s] == 1:
                        new_cons[s] += 1
                        new_off_cons[s] = 0
                        if new_cons[s] > 4:
                            if s in [0, 1] and new_cons[s] <= 5: penalty += 500
                            else: violation = True; break
                        elif new_cons[s] == 4: penalty += 50
                    else:
                        new_cons[s] = 0
                        new_offs[s] += 1
                        new_off_cons[s] += 1
                        if new_off_cons[s] >= 3:
                            penalty += 100
                            if s == 6: penalty += 200
                
                if violation: continue
                
                days_left = num_days - 1 - d
                if np.any(new_offs > req_offs): violation = True
                if np.any(new_offs + days_left < req_offs): violation = True
                if violation: continue
                
                expected = req_offs * ((d+1)/num_days)
                penalty += np.sum(np.abs(new_offs - expected)) * 10
                
                new_sched = path['sched'].copy()
                new_sched[:, d] = work_mask
                next_paths.append({'sched': new_sched, 'cons': new_cons, 'offs': new_offs, 'off_cons': new_off_cons, 'score': path['score'] + penalty})
        
        next_paths.sort(key=lambda x: x['score'])
        if not next_paths: return None
        current_paths = next_paths[:BEAM_WIDTH]
        
    best_path = current_paths[0]
    final_sched = best_path['sched']
    
    # --- 日付ヘッダーの日本語化 ---
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
                if s in roles: 
                    output_data[s, d] = roles[s]
                else:
                    caps = role_map[s]
                    if 'C' in caps: output_data[s, d] = 'C'
                    elif 'B' in caps: output_data[s, d] = 'B'
                    elif 'A' in caps: output_data[s, d] = 'A'
                    else: output_data[s, d] = 'C'
            else:
                output_data[s, d] = '×' if fixed_shifts[s, d] == '×' else '／'
        
        if is_insufficient: output_data[insufficient_row_idx, d] = "※"
            
    index_names = list(staff_df['名前']) + ["不足"]
    result_df = pd.DataFrame(output_data, columns=output_cols, index=index_names)
    return result_df

# --- スタイリング ---
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
st.title('📅 ブラウザ入力型 シフト作成ツール')

# --- サイドバー ---
with st.sidebar:
    st.header("⚙️ 設定・保存")
    
    today = datetime.date.today()
    next_month = today.replace(day=1) + datetime.timedelta(days=32)
    start_date = next_month.replace(day=1)
    next_month_end = (start_date.replace(day=1) + datetime.timedelta(days=32)).replace(day=1) - datetime.timedelta(days=1)
    
    col_d1, col_d2 = st.columns(2)
    # カレンダー入力のフォーマット指定（日本語圏向け）
    start_input = col_d1.date_input("開始日", start_date, format="YYYY/MM/DD")
    end_input = col_d2.date_input("終了日", next_month_end, format="YYYY/MM/DD")
    
    st.caption("※カレンダーの日付（月火水...）はブラウザの設定言語で表示されます")
    
    days_list = pd.date_range(start_input, end_input).tolist()
    num_days = len(days_list)
    
    st.markdown("---")
    st.subheader("💾 データの保存/読込")
    
    # 保存時に空行を除去して保存
    clean_staff_df = st.session_state.staff_df.dropna(subset=['名前'])
    clean_staff_df = clean_staff_df[clean_staff_df['名前'] != '']
    
    current_data = {
        "staff": clean_staff_df.to_dict(),
        "holidays": st.session_state.holidays_df.to_dict()
    }
    json_str = json.dumps(current_data, ensure_ascii=False)
    st.download_button("設定を保存 (JSON)", json_str, "shift_settings.json", "application/json")
    
    uploaded_json = st.file_uploader("設定を読み込む", type=["json"])
    if uploaded_json is not None:
        try:
            loaded_data = json.load(uploaded_json)
            st.session_state.staff_df = pd.DataFrame(loaded_data["staff"])
            st.session_state.holidays_df = pd.DataFrame(loaded_data["holidays"])
            st.success("設定を読み込みました！")
        except:
            st.error("読込エラー")

# --- メインエリア ---

st.markdown("### 1️⃣ スタッフ設定")
st.info("💡 **行の削除方法**: 左端の番号をクリックして行を選択し、キーボードの **Delete** キー（Macは **Fn+Delete**）を押してください。")

edited_staff_df = st.data_editor(
    st.session_state.staff_df,
    num_rows="dynamic",
    use_container_width=True,
    height=300
)
st.session_state.staff_df = edited_staff_df

# スタッフ数同期ロジック
valid_staff_count = len(edited_staff_df[edited_staff_df['名前'].notna() & (edited_staff_df['名前'] != "")])
current_holiday_rows = len(st.session_state.holidays_df)

if valid_staff_count > current_holiday_rows:
    rows_to_add = valid_staff_count - current_holiday_rows
    new_data = pd.DataFrame(False, index=range(rows_to_add), columns=st.session_state.holidays_df.columns)
    st.session_state.holidays_df = pd.concat([st.session_state.holidays_df, new_data], ignore_index=True)
elif valid_staff_count < current_holiday_rows:
    st.session_state.holidays_df = st.session_state.holidays_df.iloc[:valid_staff_count]

st.markdown("### 2️⃣ 希望休入力")
st.markdown("希望休（×）がある場合は、チェックボックスをONにしてください。")

holiday_cols = [f"Day_{i+1}" for i in range(num_days)]
display_holidays_df = st.session_state.holidays_df.reindex(columns=holiday_cols, fill_value=False)

# 名前リスト同期
valid_names = edited_staff_df[edited_staff_df['名前'].notna() & (edited_staff_df['名前'] != "")]['名前']
if len(valid_names) == len(display_holidays_df):
    display_holidays_df.index = valid_names
else:
    pass

# チェックボックス列名の日本語曜日化
edited_holidays_grid = st.data_editor(
    display_holidays_df,
    use_container_width=True,
    column_config={
        col: st.column_config.CheckboxColumn(
            f"{days_list[i].day}({['月','火','水','木','金','土','日'][days_list[i].weekday()]})", 
            default=False
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
                st.error("条件を満たすシフトが見つかりませんでした。条件を緩和してください。")
        except Exception as e:
            st.error(f"エラー: {e}")
