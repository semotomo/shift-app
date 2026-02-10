# （...上の計算ロジック部分はそのまま...）

# --- スタイリング関数（色付けのルール） ---
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
st.set_page_config(page_title="シフト作成ツール", layout="wide") # 画面を横長に使う設定

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
            
            # --- 画面表示用のデータ整形 ---
            # 見やすくするため、NaN（空白）を空文字にする
            display_df = result_df.fillna("")
            
            # スタイリング適用
            styled_df = display_df.style.map(highlight_cells)
            
            # 画面いっぱいにテーブルを表示
            st.dataframe(
                styled_df,
                use_container_width=True, # 横幅いっぱいにする
                height=600                # 縦幅を広げる
            )
            
            # CSVダウンロードボタン
            csv = result_df.to_csv(index=False, header=False).encode('utf-8-sig')
            st.download_button(
                label="📥 シフト表をダウンロード (CSV)",
                data=csv,
                file_name='完成シフト表.csv',
                mime='text/csv',
                type="primary" # ボタンを目立たせる
            )
            
        else:
            st.error("⚠️ 条件が厳しすぎて、すべてのルールを満たすシフトが組めませんでした。")
            st.markdown("条件（連勤制限や希望休）を少し緩和して、再度お試しください。")
            
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
