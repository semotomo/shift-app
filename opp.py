import streamlit as st
import pandas as pd
import numpy as np
import random
import io
import itertools

# --- ページ設定 (必ず一番最初に書く必要があります) ---
st.set_page_config(page_title="シフト作成ツール", layout="wide")

# --- 設定 ---
STAFF_ROLES_MAP = {
    0: {'A'},                
    1: {'A', 'B', 'Neko'},   
    2: {'B', 'C', 'Neko'},   
    3: {'B', 'C', 'Neko'},   
    4: {'Night'},            
    5: {'Neko', 'C'},        
    6: {'Neko', 'C'}         
}
NIGHT_IDX = 4
M1_IDX = 5
M2_IDX = 6
FULL_TIME_IDXS = [0, 1, 2, 3]

# --- 判定関数 ---
def can_cover_required_roles(staff_list):
    # 夜勤チェック
    if NIGHT_IDX in staff_list:
        if sum(1 for s in staff_list if s in FULL_TIME_IDXS) < 2: return False
    
    pool = [s for s in staff_list if s != NIGHT_IDX]
    
    # ネコチェック
    neko_fixed = None
    if M1_IDX in pool: neko_fixed = M1_IDX
    elif M2_IDX in pool: neko_fixed = M2_IDX
    
    if neko_fixed is not None:
        rem = [x for x in pool if x != neko_fixed]
        if len(rem) < 3: return False
        for p in itertools.permutations(rem, 3):
            if 'A' in STAFF_ROLES_MAP[p[0]] and 'B' in STAFF_ROLES_MAP[p[1]] and 'C' in STAFF_ROLES_MAP[p[2]]:
                return True
    else:
        if len(pool) < 4: return False
        for p in itertools.permutations(pool, 4):
            if 'Neko' in STAFF_ROLES_MAP[p[0]] and 'A' in STAFF_ROLES_MAP[p[1]] and 'B' in STAFF_ROLES_MAP[p[2]] and 'C' in STAFF_ROLES_MAP[p[3]]:
                return True
    return False

def get_possible_day_patterns(available_staff):
    patterns = []
    # サイズ3〜7の組み合わせを全列挙
    for size in range(3, 8):
        for subset in itertools.combinations(available_staff, size):
            patterns.append(subset)
    return patterns

def assign_roles_strictly(working_indices):
    assignments = {}
    if NIGHT_IDX in working_indices: assignments[NIGHT_IDX] = '〇'
    pool = [s for s in working_indices if s != NIGHT_IDX]
    neko_fixed = None
    if M1_IDX in pool: neko_fixed = M1_IDX
    elif M2_IDX in pool: neko_fixed = M2_IDX
    
    found = False
    if neko_fixed is not None:
        assignments[neko_fixed] = 'ネコ'
        rem = [x for x in pool if x != neko_fixed]
        for p in itertools.permutations(rem, 3):
            if 'A' in STAFF_ROLES_MAP[p[0]] and 'B' in STAFF_ROLES_MAP[p[1]] and 'C' in STAFF_ROLES_MAP[p[2]]:
                assignments[p[0]] = 'A'; assignments[p[1]] = 'B'; assignments[p[2]] = 'C'
                found = True
                for ex in rem:
                    if ex not in p:
                        if 'C' in STAFF_ROLES_MAP[ex]: assignments[ex] = 'C'
                        elif 'B' in STAFF_ROLES_MAP[ex]: assignments[ex] = 'B'
                break
    else:
        for p in itertools.permutations(pool, 4):
            if 'Neko' in STAFF_ROLES_MAP[p[0]] and 'A' in STAFF_ROLES_MAP[p[1]] and 'B' in STAFF_ROLES_MAP[p[2]] and 'C' in STAFF_ROLES_MAP[p[3]]:
                assignments[p[0]] = 'ネコ'; assignments[p[1]] = 'A'; assignments[p[2]] = 'B'; assignments[p[3]] = 'C'
                found = True
                for ex in pool:
                    if ex not in p:
                        if 'C' in STAFF_ROLES_MAP[ex]: assignments[ex] = 'C'
                        elif 'B' in STAFF_ROLES_MAP[ex]: assignments[ex] = 'B'
                break
    return assignments if found else {}

def solve_schedule(df):
    dates = df.iloc[1, 2:30].values
    staff_data = df.iloc[3:10, :].reset_index(drop=True)
    initial_cons = staff_data[0].astype(int).values
    req_offs = staff_data[30].astype(int).values
    fixed_shifts = staff_data.iloc[:, 2:30].values
    num_days = len(dates)
    num_staff = 7
    
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
    BEAM_WIDTH = 70
    
    for d in range(num_days):
        next_paths = []
        patterns = day_patterns[d]
        
        valid_pats = [p for p in patterns if can_cover_required_roles(p)]
        invalid_pats = [p for p in patterns if not can_cover_required_roles(p)]
        use_patterns = valid_pats[:100] + invalid_pats[:20]
        
        for path in current_paths:
            for pat in use_patterns:
                new_cons = path['cons'].copy()
                new_offs = path['offs'].copy()
                new_off_cons = path['off_cons'].copy()
                penalty = 0
                violation = False
                
                if not can_cover_required_roles(pat):
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
    
    output_df = df.copy()
    insufficient_row = [""] * 31
    insufficient_row[1] = "不足"
    
    for d in range(num_days):
        working = [s for s in range(num_staff) if final_sched[s, d] == 1]
        roles = assign_roles_strictly(working)
        is_insufficient = False
        if not can_cover_required_roles(working): is_insufficient = True
        
        for s in range(num_staff):
            r_idx = 3 + s; c_idx = 2 + d
            if s in working:
                if s in roles: output_df.iloc[r_idx, c_idx] = roles[s]
                else: 
                    if 'C' in STAFF_ROLES_MAP[s]: output_df.iloc[r_idx, c_idx] = 'C'
                    elif 'B' in STAFF_ROLES_MAP[s]: output_df.iloc[r_idx, c_idx] = 'B'
                    else: output_df.iloc[r_idx, c_idx] = '出勤'
            else:
                output_df.iloc[r_idx, c_idx] = '×' if fixed_shifts[s, d] == '×' else '／'
        
        if is_insufficient: insufficient_row[2 + d] = "※"
            
    output_df.loc[10] = insufficient_row
    return output_df

# --- スタイリング関数 ---
def highlight_cells(val):
    if val == '／':
        return 'background-color: #ffcccc; color: black' # 休み（薄い赤）
    elif val == '×':
        return 'background-color: #d9d9d9; color: gray'  # 希望休（グレー）
    elif val == '※':
        return 'background-color: #ff0000; color: white; font-weight: bold' # 不足（真っ赤）
    elif val == 'A':
        return 'background-color: #ccffff; color: black' # A（水色）
    elif val == 'B':
        return 'background-color: #ccffcc; color: black' # B（薄緑）
    elif val == 'C':
        return 'background-color: #ffffcc; color: black' # C（薄黄色）
    elif val == 'ネコ':
        return 'background-color: #ffe5cc; color: black' # ネコ（薄オレンジ）
    elif val == '〇':
        return 'background-color: #e6e6fa; color: black' # パート夜（薄紫）
    return ''

# --- Webアプリ画面 ---
st.title('📅 自動シフト作成ツール')
st.markdown("""
CSVファイルをアップロードすると、条件を満たしたシフト表を自動生成して表示します。
- **／** : 公休
- **×** : 希望休
- **※** : 人員不足（要確認）
""")

uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=['csv'])

if uploaded_file is not None:
    st.info("計算中... 最適なシフトパズルを解いています🧩")
    
    try:
        # CSV読み込み
        df_input = pd.read_csv(uploaded_file, header=None)
        
        # 計算実行
        result_df = solve_schedule(df_input)
        
        if result_df is not None:
            st.success("✨ 作成完了！")
            
            # 画面表示用のデータ整形
            display_df = result_df.fillna("")
            
            # スタイリング適用
            styled_df = display_df.style.map(highlight_cells)
            
            # 画面いっぱいにテーブルを表示
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=600
            )
            
            # CSVダウンロードボタン
            csv = result_df.to_csv(index=False, header=False).encode('utf-8-sig')
            st.download_button(
                label="📥 シフト表をダウンロード (CSV)",
                data=csv,
                file_name='完成シフト表.csv',
                mime='text/csv',
                type="primary"
            )
            
        else:
            st.error("⚠️ 条件が厳しすぎて、すべてのルールを満たすシフトが組めませんでした。")
            st.markdown("条件（連勤制限や希望休）を少し緩和して、再度お試しください。")
            
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
