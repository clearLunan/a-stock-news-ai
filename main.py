import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import akshare as ak
import warnings
import requests
import time
import urllib3
import pytz
from datetime import datetime
import os
import math

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
requests.packages.urllib3.disable_warnings()

st.set_page_config(
    page_title="AI 财经新闻概念挖掘机（同花顺版）",
    page_icon="📈",
    layout="wide"
)

API_KEY = os.getenv("ZHIPU_API_KEY")

REFRESH_INTERVAL = 120
PAGE_SIZE = 50          # 每页 50 条
MAX_PAGES = 30          # 最多缓存 30 页 = 1500 条
MAX_TOTAL = MAX_PAGES * PAGE_SIZE

def get_news():
    try:
        df = ak.stock_info_global_ths()
        required_cols = ['标题', '内容', '发布时间', '链接']
        for col in required_cols:
            if col not in df.columns:
                df[col] = '未知'
        # 取最新 100 条（接口一次最多返回这么多），后续刷新会追加
        df = df.head(100)
        return df
    except Exception as e:
        st.error(f"抓取失败: {str(e)}")
        return pd.DataFrame(columns=['标题', '内容', '发布时间', '链接'])

def get_china_time():
    china_tz = pytz.timezone('Asia/Shanghai')
    return datetime.now(china_tz).strftime("%Y-%m-%d %H:%M:%S")

def convert_to_china_time(time_str):
    if time_str in ['未知', '未知时间']:
        return time_str
    try:
        pub_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        china_tz = pytz.timezone('Asia/Shanghai')
        pub_time_china = pub_time.replace(tzinfo=pytz.UTC).astimezone(china_tz)
        return pub_time_china.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return time_str

def main():
    st.title("AI 新闻概念 & 个股挖掘工具（同花顺版）")
    st.caption("点击左侧新闻标题 → 右侧显示详情 → 点击分析按钮获取 AI 解读")

    # 初始化新闻缓存（最多 MAX_TOTAL 条）
    if 'news_df' not in st.session_state:
        st.session_state.news_df = get_news()
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    if 'last_refresh_str' not in st.session_state:
        st.session_state.last_refresh_str = get_china_time()

    # 自动刷新：整体替换缓存（新新闻在前，老新闻从后挤出）
    current_time = time.time()
    if current_time - st.session_state.last_refresh > REFRESH_INTERVAL:
        new_df = get_news()
        if not new_df.empty:
            # 合并新旧，去重（以标题+时间为唯一键）
            combined = pd.concat([new_df, st.session_state.news_df]).drop_duplicates(subset=['标题', '发布时间'], keep='first')
            # 保持最新在前
            combined = combined.sort_values(by='发布时间', ascending=False)
            # 限制总条数
            st.session_state.news_df = combined.head(MAX_TOTAL)
        st.session_state.last_refresh = current_time
        st.session_state.last_refresh_str = get_china_time()
        st.rerun()

    col_list, col_detail = st.columns([6, 4])

    with col_list:
        st.subheader("最新财经快讯（同花顺）")
        st.caption(f"上次刷新: {st.session_state.last_refresh_str}（自动每2分钟检查一次）")

        search_keyword = st.text_input("搜索新闻（关键词）", placeholder="输入标题或内容关键词，可搜索所有缓存内容...")
        search_keyword = search_keyword.strip().lower()

        # 搜索过滤所有缓存
        if search_keyword:
            filtered_df = st.session_state.news_df[
                st.session_state.news_df['标题'].str.lower().str.contains(search_keyword, na=False) |
                st.session_state.news_df['内容'].str.lower().str.contains(search_keyword, na=False)
            ]
            st.info(f"搜索到 {len(filtered_df)} 条匹配新闻（在全部 {len(st.session_state.news_df)} 条缓存中）")
        else:
            filtered_df = st.session_state.news_df

        if st.button("手动刷新新闻列表"):
            new_df = get_news()
            if not new_df.empty:
                combined = pd.concat([new_df, st.session_state.news_df]).drop_duplicates(subset=['标题', '发布时间'], keep='first')
                combined = combined.sort_values(by='发布时间', ascending=False)
                st.session_state.news_df = combined.head(MAX_TOTAL)
            st.session_state.last_refresh = time.time()
            st.session_state.last_refresh_str = get_china_time()
            st.rerun()

        # 分页
        total_items = len(filtered_df)
        total_pages = math.ceil(total_items / PAGE_SIZE) if total_items > 0 else 1

        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1

        # 限制页码
        st.session_state.current_page = max(1, min(st.session_state.current_page, total_pages))

        start_idx = (st.session_state.current_page - 1) * PAGE_SIZE
        end_idx = start_idx + PAGE_SIZE
        page_df = filtered_df.iloc[start_idx:end_idx]

        # 列表容器
        with st.container(height=900):
            if not page_df.empty:
                for idx, row in page_df.iterrows():
                    title = row.get('标题', '无标题')
                    time_str = convert_to_china_time(row.get('发布时间', '未知时间'))
                    btn_text = f"{title}   {time_str}"
                    if st.button(btn_text, key=f"news_btn_{idx}", use_container_width=True):
                        st.session_state.selected_idx = idx
                        st.rerun()
            else:
                st.info("本页无新闻")

        # 分页控件
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("上一页") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
                st.rerun()
        with col2:
            st.caption(f"第 {st.session_state.current_page} / {total_pages} 页   共 {total_items} 条新闻（缓存上限 {MAX_TOTAL} 条）")
        with col3:
            if st.button("下一页") and st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
                st.rerun()

    # 右侧详情和手动输入部分保持不变（略去重复代码，复制你原有部分即可）
    # ... (你的 col_detail 代码，包括新闻详情、分析按钮、手动输入、自动刷新设置)

if __name__ == "__main__":
    main()
