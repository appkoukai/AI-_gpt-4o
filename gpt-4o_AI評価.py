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

    def extract_keywords(self, text: str) -> List[Tuple[str, float]]:
        """キーワード抽出"""
        try:
            vectorizer = TfidfVectorizer(
                tokenizer=self.tokenize,
                token_pattern=None,
                lowercase=False,
                max_features=self.config.max_keywords
            )
            vectors = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = vectors.todense().tolist()[0]
            
            keywords = [(word, score) for word, score in zip(feature_names, scores) if score > 0]
            return sorted(keywords, key=lambda x: x[1], reverse=True)
        except Exception as e:
            st.error(f"キーワード抽出中にエラーが発生しました: {str(e)}")
            return []

class VisualizationManager:
    """可視化を管理するクラス"""
    @staticmethod
    def create_keywords_dataframe(keywords: List[Tuple[str, float]]) -> pd.DataFrame:
        """キーワードのDataFrame作成"""
        df = pd.DataFrame(keywords, columns=['単語', 'スコア'])
        df.insert(0, '順位', range(1, len(df) + 1))
        return df

    def create_heatmap(self, correlation_matrix: pd.DataFrame, keywords: List[Tuple[str, float]]) -> plt.Figure:
        """ヒートマップの作成"""
        keyword_texts = [k[0] for k in keywords[:20]]
        matrix_data = correlation_matrix.loc[keyword_texts, keyword_texts]
        
        fig = plt.figure(figsize=(12, 9))
        mask = np.triu(np.ones_like(correlation_matrix), k=0)
        
        ax = sns.heatmap(matrix_data, cmap='coolwarm', mask=mask,
                        linewidths=0.5, annot=True, fmt='.2f', cbar=True)
        
        self._setup_heatmap_axes(ax, keyword_texts)
        return fig

    @staticmethod
    def _setup_heatmap_axes(ax, keyword_texts):
        """ヒートマップの軸設定"""
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticks(np.arange(len(keyword_texts)) + 0.5)
        ax.set_xticklabels(keyword_texts, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(keyword_texts)) + 0.5)
        ax.set_yticklabels(keyword_texts, rotation=0, va='center')
        plt.tight_layout()

class AIEvaluator:
    """AI評価を管理するクラス"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def generate_evaluation(self, text: str, keywords: List[Tuple[str, float]], 
                          evaluation_points: List[str]) -> str:
        """AI評価の生成"""
        try:
            prompt = self._create_evaluation_prompt(text, keywords, evaluation_points)
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "あなたは文章分析の専門家です。"},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"AI評価生成中にエラーが発生しました: {str(e)}")
            return ""

    @staticmethod
    def _create_evaluation_prompt(text: str, keywords: List[Tuple[str, float]], 
                                evaluation_points: List[str]) -> str:
        """評価用プロンプトの作成"""
        keyword_texts = [word for word, _ in keywords]
        return f"""以下の文章の特徴語を分析して、文章の評価をして下さい:
{text}

特徴語: {', '.join(keyword_texts)}

評価する点:
{chr(10).join('- ' + point for point in evaluation_points)}"""

class StreamlitApp:
    """Streamlitアプリケーションのメインクラス"""
    def __init__(self):
        self.config = TextAnalysisConfig()
        self.analyzer = TextAnalyzer(self.config)
        self.visualizer = VisualizationManager()
        self.evaluator = AIEvaluator(self.config.model_name)
        self.initialize_session_state()

    def initialize_session_state(self):
        """セッション状態の初期化"""
        if 'last_evaluation' not in st.session_state:
            st.session_state.last_evaluation = None
        if 'user_input' not in st.session_state:
            st.session_state.user_input = ''
        if 'evaluation_points_input' not in st.session_state:
            st.session_state.evaluation_points_input = ''

    def run(self):
        """アプリケーションの実行"""
        st.title('文章評価アプリ')
        self.render_input_section()
        self.render_analysis_tabs()

    def render_input_section(self):
        """入力セクションの描画"""
        user_input = st.text_area('文章を入力', 
                                 value=st.session_state.get('user_input', ''))
        evaluation_points_input = st.text_area(
            '評価する点を入力してください（複数の場合は改行、カンマ、または空白で区切ってください）',
            value=st.session_state.get('evaluation_points_input', ''))

        if st.button('分析を実行'):
            self.perform_analysis(user_input, evaluation_points_input)

    def perform_analysis(self, user_input: str, evaluation_points_input: str):
        """分析の実行"""
        with st.spinner('分析中...'):
            keywords = self.analyzer.extract_keywords(user_input)
            keywords_data = self.visualizer.create_keywords_dataframe(keywords)
            
            # 結果をセッションステートに保存
            st.session_state.update({
                'user_input': user_input,
                'keywords': keywords,
                'keywords_data': keywords_data,
            })

    def render_analysis_tabs(self):
        """分析タブの描画"""
        tabs = st.tabs(["特徴語抽出", "共起ヒートマップ", "単語位置の可視化", "生成AIの評価"])
        
        with tabs[0]:
            self.render_keywords_tab()
        with tabs[1]:
            self.render_heatmap_tab()
        with tabs[2]:
            self.render_word_position_tab()
        with tabs[3]:
            self.render_ai_evaluation_tab()

    def render_keywords_tab(self):
        """特徴語タブの描画"""
        if 'keywords_data' in st.session_state:
            st.success('分析が完了しました')
            st.table(st.session_state.keywords_data)

    # 残りのタブ描画メソッドも同様に実装

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()