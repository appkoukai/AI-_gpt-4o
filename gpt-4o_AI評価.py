import sys
import streamlit as st
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from janome.tokenizer import Tokenizer
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import numpy as np
import math
from collections import defaultdict, Counter
import os
import re
import warnings

# OpenAI APIの設定
api_key = st.secrets["OPENAI_API_KEY"]

# Janomeトークナイザー
tokenizer = Tokenizer()

# 日本語フォントの設定
font_paths = [
    "C:\\windows\\Fonts\\YUMIN.TTF",
    "C:\\windows\\Fonts\\YUMINDB.TTF",
    "C:\\windows\\Fonts\\YUMINL.TTF"
]
for font_path in font_paths:
    if os.path.exists(font_path):
        prop = fm.FontProperties(fname=font_path)
        break
else:
    st.error("指定されたフォントが見つかりません。表示に問題がある可能性があります。")
    prop = fm.FontProperties()

# 除外する単語のリストを定義
exclusion_list = {'の', 'は', 'に', 'を', 'こと','よう','それ','もの','ん','事'}

def tokenize(text):
    tokens = tokenizer.tokenize(text)
    return [token.surface for token in tokens if token.surface not in exclusion_list and token.part_of_speech.split(',')[0] in ['名詞', '代名詞']]

def extract_keywords(text):
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    vectors = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    keywords = sorted([(word, score) for word, score in zip(feature_names, denselist[0]) if score > 0.05], key=lambda x: x[1], reverse=True)
    return keywords

def find_cooccurrences(text, keywords, window_size=30):
    cooccurrence_matrix = defaultdict(Counter)
    tokens = tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] in [k[0] for k in keywords]:
            window = tokens[i+1:i+1+window_size]
            for word in window:
                if word in [k[0] for k in keywords]:
                    cooccurrence_matrix[tokens[i]][word] += 1
                    cooccurrence_matrix[word][tokens[i]] += 1
    return cooccurrence_matrix

def draw_cooccurrence_network(cooccurrences, tfidf_scores):
    G = nx.Graph()
    for word, cooccur in cooccurrences.items():
        for co_word, weight in cooccur.items():
            G.add_edge(word, co_word, weight=weight)
    
    pos = nx.spring_layout(G)
    weights = nx.get_edge_attributes(G, 'weight')
    
    node_sizes = []
    for node in G.nodes():
        node_sizes.append(tfidf_scores.get(node, 0) * 30000)

    edge_widths = []
    for (u, v, d) in G.edges(data=True):
        log_weight = math.log(d['weight'] + 1)
        scaled_width = log_weight * 2
        edge_widths.append(scaled_width)

    colors_array = cm.Pastel2(np.linspace(0.1, 0.9, len(G.nodes())))
    node_colors = [colors_array[i % len(colors_array)] for i in range(len(G.nodes()))]

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors, node_size=node_sizes, font_size=10, font_weight='bold', font_family=prop.get_name())
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color="lightblue")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_family=prop.get_name())
    
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

def plot_word_positions(target_words, text, num_blocks=30):
    total_chars = len(text)
    block_size = total_chars // num_blocks
    fig, ax = plt.subplots(figsize=(12, 8))

    word_counts = {word: [0]*num_blocks for word in target_words}

    tokens = list(tokenizer.tokenize(text))

    char_index = 0
    block_index = 0
    for token in tokens:
        word = token.surface
        char_index += len(word)
        block_index = min(char_index // block_size, num_blocks - 1)
        if word in word_counts:
            word_counts[word][block_index] += 1

    colors = plt.cm.tab10(np.linspace(0, 1, len(target_words)))

    for i, (word, counts) in enumerate(word_counts.items()):
        if sum(counts) == 0:
            st.write(f"単語 '{word}' は文章中に見つかりませんでした。")
            continue
        max_count = max(counts)
        for j in range(num_blocks):
            if counts[j] > 0:
                height = counts[j] / max_count * 0.8  # 高さを正規化
                ax.plot([j, j], [i-height/2, i+height/2], color=colors[i], linewidth=10)

    ax.set_xlabel('文字数', fontproperties=prop)
    ax.set_title('単語の出現頻度（30分割）', fontproperties=prop)
    ax.set_yticks(range(len(target_words)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.set_yticklabels(target_words, fontproperties=prop)
    ax.invert_yaxis()

    plt.xticks(range(0, num_blocks+1, 5), [f"{i*block_size}" for i in range(0, num_blocks+1, 5)], fontproperties=prop)
    plt.yticks(fontproperties=prop)

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

        # 共起ネットワーク作成
        cooccurrences = find_cooccurrences(user_input, keywords)
        tfidf_scores = {word: score for word, score in keywords}
        cooccurrence_fig = draw_cooccurrence_network(cooccurrences, tfidf_scores)

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
tabs = st.tabs(["特徴語抽出", "共起ネットワーク描画", "単語位置の可視化", "生成AIの評価"])

with tabs[0]:
    if 'keywords_data' in st.session_state:
        st.success('分析が完了しました')
        st.table(st.session_state.keywords_data)

with tabs[1]:
    if 'cooccurrence_fig' in st.session_state:
        st.success('分析が完了しました')
        st.pyplot(st.session_state.cooccurrence_fig)

with tabs[2]:
    if 'word_position_fig' in st.session_state:
        st.success('分析が完了しました')
        st.subheader("特徴語の出現位置")
        st.pyplot(st.session_state.word_position_fig)

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