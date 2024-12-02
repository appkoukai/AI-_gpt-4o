import sys
import streamlit as st
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from janome.tokenizer import Tokenizer
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
import re
import warnings
import seaborn as sns
import matplotlib_fontja

# OpenAI APIの設定
api_key = st.secrets["OPENAI_API_KEY"]

# Janomeの初期化
tokenizer = Tokenizer()

# seabornのスタイル設定
sns.set()

# 除外する単語のリストを定義
exclusion_list = {'の', 'は', 'に', 'を', 'こと','よう','それ','もの','ん','事'}

def tokenize(text):
    tokens = []
    for token in tokenizer.tokenize(text):
        if token.surface not in exclusion_list and token.part_of_speech.split(',')[0] in ['名詞', '代名詞']:
            tokens.append(token.surface)
    return tokens
 
def extract_keywords(text):
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize, 
        token_pattern=None,
        lowercase=False,  # 大文字小文字を区別する
        max_features=100  # 最大100語に設定
    )
    vectors = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    keywords = sorted([(word, score) for word, score in zip(feature_names, denselist[0]) if score > 0], 
                     key=lambda x: x[1], 
                     reverse=True)
    return keywords[:100]  # 上位100語を返す


def calculate_pmi(text, keywords, window_size=5):
    keywords = keywords[:20]
    tokens = tokenize(text)
    keyword_set = set([k[0] for k in keywords])
    
    # 単語の個別出現回数
    word_counts = {k[0]: 0 for k in keywords}
    # 共起回数
    cooccurrence_counts = {k[0]: {k2[0]: 0 for k2 in keywords} for k in keywords}
    # 総ウィンドウ数
    total_windows = len(tokens) - window_size + 1
    
    # 単語カウントと共起カウント
    for i in range(len(tokens)):
        if tokens[i] in keyword_set:
            word_counts[tokens[i]] += 1
            
        if i < total_windows:
            window = tokens[i:i+window_size]
            window_words = set(w for w in window if w in keyword_set)
            for w1 in window_words:
                for w2 in window_words:
                    if w1 != w2:
                        cooccurrence_counts[w1][w2] += 1
    
    # PMI行列の計算
    pmi_matrix = {}
    for w1 in keyword_set:
        pmi_matrix[w1] = {}
        for w2 in keyword_set:
            if w1 != w2:
                p_w1 = word_counts[w1] / total_windows
                p_w2 = word_counts[w2] / total_windows
                p_w1w2 = cooccurrence_counts[w1][w2] / total_windows
                
                if p_w1w2 > 0:
                    pmi = np.log2(p_w1w2 / (p_w1 * p_w2))
                else:
                    pmi = 0
                pmi_matrix[w1][w2] = pmi
    
    return pd.DataFrame(pmi_matrix)

def create_cooccurrence_heatmap(correlation_matrix, keywords):
     # キーワードのリストを作成
    keyword_texts = [k[0] for k in keywords[:20]]  # Top 20 keywords
    
    # correlation_matrixから必要な部分だけを抽出
    matrix_data = correlation_matrix.loc[keyword_texts, keyword_texts]
    fig = plt.figure(figsize=(12, 9))
    
    # 上三角を除去
    mask = np.triu(np.ones_like(correlation_matrix), k=0)
    
    # ヒートマップを描画
    ax = sns.heatmap(matrix_data, cmap='coolwarm', mask=mask, 
                     linewidths=0.5, annot=True, fmt='.2f', cbar=True)
    
    plt.title('特徴語の共起相関ヒートマップ (上位20語)', fontsize=16)
    
    # 軸ラベルを削除
    ax.set_xticks([])
    ax.set_yticks([])
    
    keyword_texts = [k[0] for k in keywords[:20]]  # Top 20 keywords
    
    # x軸のラベルを配置
    ax.set_xticks(np.arange(len(keyword_texts)) + 0.5)
    ax.set_xticklabels(keyword_texts, rotation=45, ha='right')
    
    # y軸のラベルを配置
    ax.set_yticks(np.arange(len(keyword_texts)) + 0.5)
    ax.set_yticklabels(keyword_texts, rotation=0, va='center')
    
    plt.tight_layout()
    return fig

def parse_evaluation_points(input_text):
    split_pattern = r'[　 ,、。\n]+'
    points = re.split(split_pattern, input_text)
    return [point.strip() for point in points if point.strip()]

def generate_initial_evaluation(user_input, keywords, evaluation_points):
    prompt = f"""以下の文章の特徴語を分析して、文章の評価をして下さい:
{user_input}

特徴語: {', '.join([word for word, score in keywords])}

評価する点:
{chr(10).join('- ' + point for point in evaluation_points)}"""
    
    client = openai
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたは文章分析の専門家です。与えられた文章を分析し、評価を行ってください。"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def generate_additional_evaluation(previous_evaluation, additional_points, corrections):
    prompt = f"""前回の評価:
{previous_evaluation}

追加で評価する点:
{additional_points}

修正が必要な点:
{corrections}

上記の情報を踏まえて、より詳細で正確な評価を行ってください。
特に修正が必要な点については、どのように評価を修正すべきか具体的に説明してください。"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたは文章分析の専門家です。前回の評価を踏まえて、追加の評価と修正を行ってください。"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def plot_word_positions(target_words, text, num_blocks=20):
    target_words = target_words[:15]
    total_chars = len(text)
    block_size = total_chars // num_blocks
    fig, ax = plt.subplots(figsize=(12, 8))

    # 格子を描画
    for i in range(num_blocks + 1):
        ax.axvline(x=i, color='gray', linestyle=':', alpha=0.3)
    for i in range(len(target_words) + 1):
        ax.axhline(y=i, color='gray', linestyle=':', alpha=0.3)

    word_counts = {word: [0]*num_blocks for word in target_words}

    # Janomeで形態素解析して位置をカウント
    char_count = 0
    for token in tokenizer.tokenize(text):
        word = token.surface
        char_count += len(word)
        block_index = min(char_count // block_size, num_blocks - 1)
        if word in word_counts:
            word_counts[word][block_index] += 1
    colors = plt.cm.tab10(np.linspace(0, 1, len(target_words)))

    # 円で出現率を表現
    for i, (word, counts) in enumerate(word_counts.items()):
        if sum(counts) == 0:
            st.write(f"単語 '{word}' は文章中に見つかりませんでした。")
            continue
        max_count = max(max(counts), 1)  # ゼロ除算を防ぐ
        for j in range(num_blocks):
            if counts[j] > 0:
                size = counts[j] / max_count * 1200  # サイズを調整
                ax.scatter(j + 0.5, i, s=size, color=colors[i], alpha=0.6)

    ax.set_xlabel('文字数',)
    ax.set_title('特徴語の出現位置')
    ax.set_yticks(range(len(target_words)))
    ax.invert_yaxis()

    # X軸のラベルを設定
    xticks = np.linspace(0, num_blocks, 5)
    xtick_labels = [f"{int(x * total_chars / num_blocks)}" for x in xticks]
    plt.xticks(xticks, xtick_labels)
    plt.yticks()

    plt.tight_layout()
    return fig

# Streamlit UIの構築
st.title('文章評価アプリ')

# セッションステートの初期化
if 'last_evaluation' not in st.session_state:
    st.session_state.last_evaluation = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''
if 'evaluation_points_input' not in st.session_state:
    st.session_state.evaluation_points_input = ''

# ユーザー入力
user_input = st.text_area('文章を入力', value=st.session_state.get('user_input', ''))
evaluation_points_input = st.text_area('評価する点を入力してください（複数の場合は改行、カンマ、または空白で区切ってください）', value=st.session_state.get('evaluation_points_input', ''))
evaluation_points = parse_evaluation_points(evaluation_points_input)

# 初期分析の実行
if st.button('分析を実行'):
    with st.spinner('分析中...'):
        # キーワード抽出
        keywords = extract_keywords(user_input)
        keywords_data = pd.DataFrame(keywords, columns=['単語', 'スコア'])
        keywords_data = keywords_data.sort_values(by='スコア', ascending=False).reset_index(drop=True)
        keywords_data.insert(0, '順位', range(1, len(keywords_data) + 1))

        # 共起ヒートマップ作成
        correlation_matrix = calculate_pmi(user_input, keywords)
        cooccurrence_fig = create_cooccurrence_heatmap(correlation_matrix, keywords)
        
        # 単語位置の可視化
        top_keywords = [word for word, _ in keywords[:10]]  # 上位10個の特徴語を使用
        word_position_fig = plot_word_positions(top_keywords, user_input)

        # セッションステートに保存
        st.session_state.user_input = user_input
        st.session_state.keywords = keywords
        st.session_state.keywords_data = keywords_data
        st.session_state.cooccurrence_fig = cooccurrence_fig
        st.session_state.word_position_fig = word_position_fig

# タブ分け
tabs = st.tabs(["特徴語抽出", "共起ヒートマップ", "単語位置の可視化", "生成AIの評価"])

with tabs[0]:
    if 'keywords_data' in st.session_state:
        st.success('分析が完了しました')
        st.table(st.session_state.keywords_data)

with tabs[1]:
    if 'keywords_data' in st.session_state:
        st.success('分析が完了しました')
        
        # 特徴語の選択機能を追加
        all_keywords = [word for word, _ in st.session_state.keywords]
        default_selection = all_keywords[:20]
        selected_keywords_heatmap = st.multiselect(
            '共起ヒートマップに表示する特徴語を選択してください（最大20語）',
            all_keywords,
            default=default_selection
        )
        
        if len(selected_keywords_heatmap) > 20:
            st.warning('20語以上選択されています。上位20語のみ表示します。')
            selected_keywords_heatmap = selected_keywords_heatmap[:20]
        
        if selected_keywords_heatmap:
            # 選択された特徴語でヒートマップを再生成
            selected_keywords_with_scores = [
                (word, score) for word, score in st.session_state.keywords 
                if word in selected_keywords_heatmap
            ]
            correlation_matrix = calculate_pmi(st.session_state.user_input, selected_keywords_with_scores)
            new_heatmap_fig = create_cooccurrence_heatmap(correlation_matrix, selected_keywords_with_scores)
            st.pyplot(new_heatmap_fig)

with tabs[2]:
    if 'keywords_data' in st.session_state:
        st.success('分析が完了しました')
        st.subheader("特徴語の出現位置")
        
        # 特徴語の選択機能を追加
        all_keywords = [word for word, _ in st.session_state.keywords]
        default_selection = all_keywords[:15]
        selected_keywords_plot = st.multiselect(
            '出現位置プロットに表示する特徴語を選択してください（最大15語）',
            all_keywords,
            default=default_selection
        )
        
        if len(selected_keywords_plot) > 15:
            st.warning('15語以上選択されています。上位15語のみ表示します。')
            selected_keywords_plot = selected_keywords_plot[:15]
        
        if selected_keywords_plot:
            # 選択された特徴語でプロットを再生成
            new_position_fig = plot_word_positions(selected_keywords_plot, st.session_state.user_input)
            st.pyplot(new_position_fig)

with tabs[3]:
    if st.button('生成AIの評価を実行'):
        with st.spinner('評価生成中...'):
            # 初期評価の生成
            evaluation = generate_initial_evaluation(user_input, st.session_state.keywords, evaluation_points)
        
        st.success('評価が完了しました')
        st.write("評価ポイント:")
        for point in evaluation_points:
            st.write(f"- {point}")
        
        st.write("AI評価:")
        st.write(evaluation)

        # セッションステートに保存
        st.session_state.last_evaluation = evaluation

# 追加評価と修正のセクション
if st.session_state.last_evaluation:
    st.markdown("---")
    st.subheader("追加評価と修正")
    
    additional_points = st.text_area("より詳しく評価したい点があれば入力してください")
    corrections = st.text_area("AIの評価で間違っている点や修正が必要な点があれば入力してください")
    
    if st.button('追加評価・修正を実行'):
        with st.spinner('追加分析中...'):
            updated_evaluation = generate_additional_evaluation(
                st.session_state.last_evaluation,
                additional_points,
                corrections
            )
        
        st.success('追加分析が完了しました')
        st.write("更新されたAI評価:")
        st.write(updated_evaluation)
        
        # 新しい評価結果を保存
        st.session_state.last_evaluation = updated_evaluation
