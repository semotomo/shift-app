import streamlit as st
import pandas as pd
import numpy as np
import random
import io
import itertools

# --- 設定 ---
# 0: A1, 1: A2, 2: B1, 3: B2, 4: Night, 5: M1, 6: M2
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

def can_cover_abc(staff_list):
    for p in itertools.permutations(staff_list, 3):
        if 'A' in STAFF_ROLES_MAP[p[0]] and 'B' in STAFF_ROLES_MAP[p[1]] and 'C' in STAFF_ROLES_MAP[p[2]]:
            return True
    return False

def can_cover_abc_neko(staff_list):
    for p in itertools.permutations(staff_list, 4):
        if 'Neko' in STAFF_ROLES_MAP[p[0]] and 'A' in STAFF_ROLES_MAP[p[1]] and 'B' in STAFF_ROLES_MAP[p[2]] and 'C' in STAFF_ROLES_MAP[p[3]]:
            return True
    return False

def get_valid_day_patterns(available_staff):
    valid_patterns = []
    for size in range(4, 7):
        for subset in itertools.combinations(available_staff, size):
            if NIGHT_IDX in subset:
                if sum(1 for s in subset if s in FULL_TIME_IDXS) < 2: continue
            neko = None
            if M1_IDX in subset: neko = M1_IDX
            elif M2_IDX in subset: neko = M2_IDX
            
            if neko is not None:
                rem = [s for s in subset if s != neko and s != NIGHT_IDX]
                if can_cover_abc(rem): valid_patterns.append(subset)
            else:
                pool = [s for s in subset if s != NIGHT_IDX]
                if can_cover_abc_neko(pool): valid_patterns.append(subset)
    return valid_patterns

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
        rem_pool = [x for x in pool if x != neko_fixed]
        for p in itertools.permutations(rem_pool, 3):
            if 'A' in STAFF_ROLES_MAP[p[0]] and 'B' in STAFF_ROLES_MAP[p[1]] and 'C' in STAFF_ROLES_MAP[p[2]]:
                assignments[p[0]] = 'A'; assignments[p[1]] = 'B'; assignments[p[2]] = 'C'
                found = True
                for extra in rem_pool:
                    if extra not in p:
                        if 'C' in STAFF_ROLES_MAP[extra]: assignments[extra] = 'C'
                        elif 'B' in STAFF_ROLES_MAP[extra]: assignments[extra] = 'B'
                break
    else:
        for p in itertools.permutations(pool, 4):
            if 'Neko' in STAFF_ROLES_MAP[p[0]] and 'A' in STAFF_ROLES_MAP[p[1]] and 'B' in STAFF_ROLES_MAP[p[2]] and 'C' in STAFF_ROLES_MAP[p[3]]:
                assignments[p[0]] = 'ネコ'; assignments[p[1]] = 'A'; assignments[p[2]] = 'B'; assignments[p[3]] = 'C'
                found = True
                for extra in pool:
                    if extra not in p:
                        if 'C' in STAFF_ROLES_MAP[extra]: assignments[extra] = 'C'
                        elif 'B' in STAFF_ROLES_MAP[extra]: assignments[extra] = 'B'
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
        pats = get_valid_day_patterns(avail)
        if not pats: return None
        random.shuffle(pats)
        day_patterns.append(pats)

    # 状態: sched, cons(連勤), offs(休日数), off_cons(連休), score
    current_paths = [{
        'sched': np.zeros((num_staff, num_days), dtype=int), 
        'cons': initial_cons.copy(), 
        'offs': np.zeros(num_staff, dtype=int), 
        'off_cons': np.zeros(num_staff, dtype=int),
        'score': 0
    }]
    BEAM_WIDTH = 50
    
    for d in range(num_days):
        next_paths = []
        patterns = day_patterns[d]
        if len(patterns) > 150: patterns = random.sample(patterns, 150)
        
        for path in current_paths:
            for pat in patterns:
                new_cons = path['cons'].copy()
                new_offs = path['offs'].copy()
                new_off_cons = path['off_cons'].copy()
                penalty = 0
                violation = False
                
                work_mask = np.zeros(num_staff, dtype=int)
                for s in pat: work_mask[s] = 1
                
                for s in range(num_staff):
                    if work_mask[s] == 1:
                        new_cons[s] += 1
                        new_off_cons[s] = 0
                        # 連勤制限 (厳しめ: 3連勤推奨)
                        if new_cons[s] > 4: 
                            if s in [0, 1] and new_cons[s] <= 5: penalty += 500
                            else: violation = True; break
                        elif new_cons[s] == 4: penalty += 50 # 4連勤は少し避ける
                    else:
                        new_cons[s] = 0
                        new_offs[s] += 1
                        new_off_cons[s] += 1
                        # 連休制限 (3連休以上を避ける＝分散させる)
                        if new_off_cons[s] >= 3: 
                            if s == 6: penalty += 500 # 朝2は特に分散
                            else: penalty += 100

                if violation: continue
                
                # 休日数のペース配分チェック（分散のため）
                days_left = num_days - 1 - d
                if np.any(new_offs > req_offs): penalty += 200
                if np.any(new_offs + days_left < req_offs): penalty += 2000
                
                # 理想的な休日ペースからの乖離をペナルティ化
                expected_offs = req_offs * ((d + 1) / num_days)
                diffs = np.abs(new_offs - expected_offs)
                penalty += np.sum(diffs) * 10

                new_sched = path['sched'].copy()
                new_sched[:, d] = work_mask
                next_paths.append({'sched': new_sched, 'cons': new_cons, 'offs': new_offs, 'off_cons': new_off_cons, 'score': path['score'] + penalty})
        
        next_paths.sort(key=lambda x: x['score'])
        if not next_paths: return None
        current_paths = next_paths[:BEAM_WIDTH]
        
    best_path = current_paths[0]
    final_sched = best_path['sched']
    output_df = df.copy()
    
    for d in range(num_days):
        working = [s for s in range(num_staff) if final_sched[s, d] == 1]
        roles = assign_roles_strictly(working)
        for s in range(num_staff):
            r_idx = 3 + s; c_idx = 2 + d
            if s in working: output_df.iloc[r_idx, c_idx] = roles.get(s, 'C')
            else: output_df.iloc[r_idx, c_idx] = '×' if fixed_shifts[s, d] == '×' else '／'
    return output_df

# --- Webアプリ画面 ---
st.title('自動シフト作成ツール (バランス調整版)')
st.markdown("CSVファイルをアップロードすると、連勤を抑え、休みを分散させたシフト表を自動生成します。")

uploaded_file = st.file_uploader("CSVファイルをアップロード", type=['csv'])

if uploaded_file is not None:
    st.info("計算中... バランスを調整しています")
    try:
        df_input = pd.read_csv(uploaded_file, header=None)
        result_df = solve_schedule(df_input)
        
        if result_df is not None:
            st.success("作成完了！")
            csv = result_df.to_csv(index=False, header=False).encode('utf-8-sig')
            st.download_button(
                label="シフト表をダウンロード (CSV)",
                data=csv,
                file_name='バランス調整版シフト表.csv',
                mime='text/csv',
            )
        else:
            st.error("条件が厳しすぎてシフトが組めませんでした。")
            
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
