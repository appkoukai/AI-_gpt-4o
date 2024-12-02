import sys
import streamlit as st
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from janome.tokenizer import Tokenizer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import matplotlib_fontja
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import joblib

# 以下のモジュールがインストールされていない場合はインストールしてください
# !pip install streamlit openai scikit-learn janome pandas matplotlib seaborn matplotlib-fontja

@dataclass
class TextAnalysisConfig:
    """設定値を管理するクラス"""
    exclusion_list: set = frozenset({'の', 'は', 'に', 'を', 'こと','よう','それ','もの','ん','事'})
    max_keywords: int = 100
    max_display_keywords: int = 20
    window_size: int = 5
    model_name: str = "gpt-4o-mini"

class TextAnalyzer:
    """テキスト分析を行うクラス"""
    def __init__(self, config: TextAnalysisConfig):
        self.config = config
        self.tokenizer = Tokenizer()
        self.setup_visualization()
        
    def setup_visualization(self):
        """可視化の設定"""
        sns.set(font="IPAexGothic")
        
    def tokenize(self, text: str) -> List[str]:
        """テキストのトークン化"""
        tokens = []
        for token in self.tokenizer.tokenize(text):
            if (token.surface not in self.config.exclusion_list and 
                token.part_of_speech.split(',')[0] in ['名詞', '代名詞']):
                tokens.append(token.surface)
        return tokens

    # 特徴語抽出
def extract_keywords(self, text: str) -> List[Tuple[str, float, int]]:
    """キーワード抽出"""
    try:
        # 事前学習済みのTF-IDFベクトライザーを読み込む
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        
        # テキストをベクトル化
        vectors = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = vectors.todense().tolist()[0]
        
        # 出現頻度のカウント
        word_freq = {}
        tokens = self.tokenize(text)
        for token in tokens:
            if token in feature_names:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        keywords = []
        for word, score in zip(feature_names, scores):
            if score > 0:
                freq = word_freq.get(word, 0)
                keywords.append((word, score, freq))
        
        # スコアで降順ソート
        keywords.sort(key=lambda x: x[1], reverse=True)
        return keywords
    except Exception as e:
        st.error(f"キーワード抽出中にエラーが発生しました: {str(e)}")
        return []

    def calculate_pmi(self, text: str, keywords: List[Tuple[str, float, int]]) -> pd.DataFrame:
        """PMI行列の計算"""
        try:
            top_keywords = keywords[:20]
            tokens = self.tokenize(text)
            keyword_set = set([k[0] for k in top_keywords])
            
            # 単語の個別出現回数
            word_counts = {k[0]: 0 for k in top_keywords}
            # 共起回数
            cooccurrence_counts = {k[0]: {k2[0]: 0 for k2 in top_keywords} for k in top_keywords}
            # 総ウィンドウ数
            total_windows = len(tokens) - self.config.window_size + 1
            total_windows = total_windows if total_windows > 0 else 1  # 防御的プログラミング
            
            # 単語カウントと共起カウント
            for i in range(len(tokens)):
                if tokens[i] in keyword_set:
                    word_counts[tokens[i]] += 1
                    
                if i < total_windows:
                    window = tokens[i:i+self.config.window_size]
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
        except Exception as e:
            st.error(f"PMI計算中にエラーが発生しました: {str(e)}")
            return pd.DataFrame()

    def plot_word_positions(self, target_words: List[str], text: str, num_blocks=20) -> plt.Figure:
        """特徴語の出現位置をプロット"""
        try:
            target_words = target_words[:15]
            total_chars = len(text)
            block_size = total_chars // num_blocks if num_blocks > 0 else 1
            fig, ax = plt.subplots(figsize=(12, 8))
    
            # 格子を描画
            for i in range(num_blocks + 1):
                ax.axvline(x=i, color='gray', linestyle=':', alpha=0.3)
            for i in range(len(target_words) + 1):
                ax.axhline(y=i, color='gray', linestyle=':', alpha=0.3)
    
            word_counts = {word: [0]*num_blocks for word in target_words}
    
            # Janomeで形態素解析して位置をカウント
            char_count = 0
            tokens = self.tokenize(text)
            for token in self.tokenizer.tokenize(text):
                word = token.surface
                char_count += len(word)
                block_index = min(char_count // block_size, num_blocks - 1) if block_size else 0
                if word in word_counts:
                    word_counts[word][block_index] += 1
            colors = plt.cm.tab10(np.linspace(0, 1, len(target_words)))
    
            # 円で出現率を表現
            for i, (word, counts) in enumerate(word_counts.items()):
                if sum(counts) == 0:
                    st.write(f"単語 '{word}' は文章中に見つかりませんでした。")
                    continue
                max_count = max(counts) if max(counts) > 0 else 1  # ゼロ除算を防ぐ
                for j in range(num_blocks):
                    if counts[j] > 0:
                        size = (counts[j] / max_count) * 1200  # サイズを調整
                        ax.scatter(j + 0.5, i, s=size, color=colors[i], alpha=0.6)
    
            ax.set_xlabel('文字数')
            ax.set_title('特徴語の出現位置')
            ax.set_yticks(range(len(target_words)))
            ax.set_yticklabels(target_words)
            ax.invert_yaxis()
    
            # X軸のラベルを設定
            xticks = np.linspace(0, num_blocks, 5)
            xtick_labels = [f"{int(x * total_chars / num_blocks)}" for x in xticks]
            plt.xticks(xticks, xtick_labels)
    
            plt.yticks()
    
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"単語位置のプロット中にエラーが発生しました: {str(e)}")
            return plt.Figure()

class VisualizationManager:
    """可視化を管理するクラス"""
    @staticmethod
    def create_keywords_dataframe(keywords: List[Tuple[str, float, int]]) -> pd.DataFrame:
        """キーワードのDataFrame作成"""
        df = pd.DataFrame(keywords, columns=['単語', 'スコア', '出現回数'])
        df.insert(0, '順位', range(1, len(df) + 1))
        return df

    @staticmethod
    def create_heatmap(correlation_matrix: pd.DataFrame, keywords: List[Tuple[str, float, int]]) -> plt.Figure:
        """ヒートマップの作成"""
        keyword_texts = [k[0] for k in keywords[:20]]
        matrix_data = correlation_matrix.loc[keyword_texts, keyword_texts]
        
        fig = plt.figure(figsize=(12, 9))
        mask = np.triu(np.ones_like(matrix_data, dtype=bool))
        
        ax = sns.heatmap(matrix_data, cmap='coolwarm', mask=mask,
                        linewidths=0.5, annot=True, fmt='.2f', cbar=True)
        
        # ヒートマップの軸設定
        ax.set_xticks(np.arange(len(keyword_texts)) + 0.5)
        ax.set_xticklabels(keyword_texts, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(keyword_texts)) + 0.5)
        ax.set_yticklabels(keyword_texts, rotation=0, va='center')
        plt.title('特徴語の共起相関ヒートマップ (上位20語)', fontsize=16)
        plt.tight_layout()
        return fig

class AIEvaluator:
    """AI評価を管理するクラス"""
    def __init__(self, config: TextAnalysisConfig):
        self.config = config
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        
    def generate_initial_evaluation(self, text: str, keywords: List[Tuple[str, float, int]], 
                                    evaluation_points: List[str]) -> str:
        """初期評価の生成"""
        try:
            prompt = self._create_evaluation_prompt(text, keywords, evaluation_points)
            response = openai.ChatCompletion.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "あなたは文章分析の専門家です。与えられた文章を分析し、評価を行ってください。"},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"AI評価生成中にエラーが発生しました: {str(e)}")
            return ""

    def generate_additional_evaluation(self, previous_evaluation: str, additional_points: str, 
                                       corrections: str) -> str:
        """追加評価と修正の生成"""
        try:
            prompt = self._create_additional_evaluation_prompt(previous_evaluation, additional_points, corrections)
            response = openai.ChatCompletion.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "あなたは文章分析の専門家です。前回の評価を踏まえて、追加の評価と修正を行ってください。"},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"追加評価生成中にエラーが発生しました: {str(e)}")
            return ""

    @staticmethod
    def _create_evaluation_prompt(text: str, keywords: List[Tuple[str, float, int]], 
                                 evaluation_points: List[str]) -> str:
        """評価用プロンプトの作成"""
        keyword_texts = [word for word, _, _ in keywords]
        return f"""以下の文章の特徴語を分析して、文章の評価をして下さい:
{text}

特徴語: {', '.join(keyword_texts)}

評価する点:
{chr(10).join('- ' + point for point in evaluation_points)}"""

    @staticmethod
    def _create_additional_evaluation_prompt(previous_evaluation: str, additional_points: str, 
                                            corrections: str) -> str:
        """追加評価用プロンプトの作成"""
        return f"""前回の評価:
{previous_evaluation}

追加で評価する点:
{additional_points}

修正が必要な点:
{corrections}

上記の情報を踏まえて、より詳細で正確な評価を行ってください。
特に修正が必要な点については、どのように評価を修正すべきか具体的に説明してください。"""

class StreamlitApp:
    """Streamlitアプリケーションのメインクラス"""
    def __init__(self):
        self.config = TextAnalysisConfig()
        self.analyzer = TextAnalyzer(self.config)
        self.visualizer = VisualizationManager()
        self.evaluator = AIEvaluator(self.config)
        self.initialize_session_state()

    def initialize_session_state(self):
        """セッション状態の初期化"""
        if 'last_evaluation' not in st.session_state:
            st.session_state.last_evaluation = None
        if 'user_input' not in st.session_state:
            st.session_state.user_input = ''
        if 'evaluation_points_input' not in st.session_state:
            st.session_state.evaluation_points_input = ''
        if 'keywords' not in st.session_state:
            st.session_state.keywords = []
        if 'keywords_data' not in st.session_state:
            st.session_state.keywords_data = pd.DataFrame()
        if 'cooccurrence_fig' not in st.session_state:
            st.session_state.cooccurrence_fig = None
        if 'word_position_fig' not in st.session_state:
            st.session_state.word_position_fig = None

    def run(self):
        """アプリケーションの実行"""
        st.title('文章評価アプリ')
        self.render_input_section()
        self.render_analysis_tabs()
        self.render_evaluation_sections()

    def render_input_section(self):
        """入力セクションの描画"""
        user_input = st.text_area('文章を入力', 
                                   value=st.session_state.get('user_input', ''))
        evaluation_points_input = st.text_area(
            '評価する点を入力してください（複数の場合は改行、カンマ、または空白で区切ってください）',
            value=st.session_state.get('evaluation_points_input', '')
        )

        if st.button('分析を実行'):
            self.perform_analysis(user_input, evaluation_points_input)

    def perform_analysis(self, user_input: str, evaluation_points_input: str):
        """分析の実行"""
        with st.spinner('分析中...'):
            evaluation_points = self.parse_evaluation_points(evaluation_points_input)
            keywords = self.analyzer.extract_keywords(user_input)
            keywords_data = self.visualizer.create_keywords_dataframe(keywords)
            correlation_matrix = self.analyzer.calculate_pmi(user_input, keywords)
            cooccurrence_fig = self.visualizer.create_heatmap(correlation_matrix, keywords)
            
            top_keywords = [word for word, _, _ in keywords[:10]]  # 上位10個の特徴語を使用
            word_position_fig = self.analyzer.plot_word_positions(top_keywords, user_input)
            
            # 結果をセッションステートに保存
            st.session_state.update({
                'user_input': user_input,
                'evaluation_points_input': evaluation_points_input,
                'keywords': keywords,
                'keywords_data': keywords_data,
                'cooccurrence_fig': cooccurrence_fig,
                'word_position_fig': word_position_fig,
                'evaluation_points': evaluation_points
            })

    @staticmethod
    def parse_evaluation_points(input_text: str) -> List[str]:
        """評価ポイントのパース"""
        split_pattern = r'[　 ,、。\n]+'
        points = re.split(split_pattern, input_text)
        return [point.strip() for point in points if point.strip()]

    def render_analysis_tabs(self):
        """分析タブの描画"""
        if not st.session_state.keywords_data.empty:
            tabs = st.tabs(["特徴語抽出", "共起ヒートマップ", "単語位置の可視化"])
            
            with tabs[0]:
                self.render_keywords_tab()
            with tabs[1]:
                self.render_heatmap_tab()
            with tabs[2]:
                self.render_word_position_tab()

    def render_keywords_tab(self):
        """特徴語タブの描画"""
        st.success('分析が完了しました')
        st.table(st.session_state.keywords_data)

    def render_heatmap_tab(self):
        """共起ヒートマップタブの描画"""
        st.success('分析が完了しました')
        
        # 特徴語の選択機能を追加
        all_keywords = [word for word, _, _ in st.session_state.keywords]
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
                (word, score, freq) for word, score, freq in st.session_state.keywords 
                if word in selected_keywords_heatmap
            ]
            correlation_matrix = self.analyzer.calculate_pmi(st.session_state.user_input, selected_keywords_with_scores)
            new_heatmap_fig = self.visualizer.create_heatmap(correlation_matrix, selected_keywords_with_scores)
            st.pyplot(new_heatmap_fig)

    def render_word_position_tab(self):
        """単語位置の可視化タブの描画"""
        st.success('分析が完了しました')
        st.subheader("特徴語の出現位置")
        
        # 特徴語の選択機能を追加
        all_keywords = [word for word, _, _ in st.session_state.keywords]
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
            new_position_fig = self.analyzer.plot_word_positions(selected_keywords_plot, st.session_state.user_input)
            st.pyplot(new_position_fig)

    def render_evaluation_sections(self):
        """生成AIの評価および追加評価セクションの描画"""
        tabs = st.tabs(["生成AIの評価"])
        
        with tabs[0]:
            if st.button('生成AIの評価を実行'):
                self.perform_initial_evaluation()

            if st.session_state.last_evaluation:
                st.markdown("---")
                st.subheader("AI評価")
                st.write(st.session_state.last_evaluation)

                st.markdown("---")
                st.subheader("追加評価と修正")
                
                additional_points = st.text_area("より詳しく評価したい点があれば入力してください")
                corrections = st.text_area("AIの評価で間違っている点や修正が必要な点があれば入力してください")
                
                if st.button('追加評価・修正を実行'):
                    self.perform_additional_evaluation(additional_points, corrections)

    def perform_initial_evaluation(self):
        """初期評価の実行"""
        with st.spinner('評価生成中...'):
            evaluation = self.evaluator.generate_initial_evaluation(
                st.session_state.user_input,
                st.session_state.keywords,
                st.session_state.get('evaluation_points', [])
            )
            if evaluation:
                st.success('評価が完了しました')
                st.session_state.last_evaluation = evaluation

    def perform_additional_evaluation(self, additional_points: str, corrections: str):
        """追加評価と修正の実行"""
        with st.spinner('追加分析中...'):
            updated_evaluation = self.evaluator.generate_additional_evaluation(
                st.session_state.last_evaluation,
                additional_points,
                corrections
            )
            if updated_evaluation:
                st.success('追加分析が完了しました')
                st.session_state.last_evaluation = updated_evaluation

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
