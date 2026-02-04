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

st.set_page_config(page_title="AI 财经新闻概念挖掘机", page_icon="📈", layout="wide")

API_KEY = os.getenv("ZHIPU_API_KEY")

REFRESH_INTERVAL = 120
PAGE_SIZE = 50
ITEMS_PER_COLUMN = 25
MAX_TOTAL = 1500

# 极致压缩按钮样式
st.markdown("""
    <style>
    .stButton > button {
        font-size: 12px !important;
        padding: 4px 8px !important;
        line-height: 1.0 !important;
        height: 50px !important;
        margin-bottom: 2px !important;
        white-space: normal !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        display: block !important;
    }
    .stButton {
        margin-bottom: 2px !important;
    }
    </style>
    """, unsafe_allow_html=True)

def get_news():
    try:
        df = ak.stock_info_global_ths()
        required_cols = ['标题', '内容', '发布时间', '链接']
        for col in required_cols:
            if col not in df.columns:
                df[col] = '未知'
        return df.head(200)
    except:
        return pd.DataFrame(columns=['标题', '内容', '发布时间', '链接'])

def get_china_time():
    return datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S")

def convert_to_china_time(time_str):
    if not time_str or time_str in ['未知', '未知时间']:
        return time_str
    try:
        pub_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return pub_time.replace(tzinfo=pytz.UTC).astimezone(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return time_str

def main():
    st.title("AI 新闻概念 & 个股挖掘工具（同花顺版）")
    st.caption("点击新闻标题查看详情和AI分析")

    if 'news_df' not in st.session_state:
        st.session_state.news_df = get_news()
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    if 'last_refresh_str' not in st.session_state:
        st.session_state.last_refresh_str = get_china_time()

    # 自动刷新：追加新新闻，保留旧新闻
    current_time = time.time()
    if current_time - st.session_state.last_refresh > REFRESH_INTERVAL:
        new_df = get_news()
        if not new_df.empty:
            combined = pd.concat([new_df, st.session_state.news_df]).drop_duplicates(subset=['标题', '发布时间'], keep='first')
            combined = combined.sort_values(by='发布时间', ascending=False)
            st.session_state.news_df = combined.head(MAX_TOTAL)
        st.session_state.last_refresh = current_time
        st.session_state.last_refresh_str = get_china_time()
        st.rerun()

    col_list, col_detail = st.columns([7, 3])

    with col_list:
        st.subheader("最新财经快讯")
        st.caption(f"上次刷新: {st.session_state.last_refresh_str}（每2分钟自动）")

        search_keyword = st.text_input("搜索（全缓存搜索）", placeholder="输入关键词...")
        search_keyword = search_keyword.strip().lower()

        if search_keyword:
            filtered_df = st.session_state.news_df[
                st.session_state.news_df['标题'].str.lower().str.contains(search_keyword, na=False) |
                st.session_state.news_df['内容'].str.lower().str.contains(search_keyword, na=False)
            ]
            st.info(f"找到 {len(filtered_df)} 条（缓存 {len(st.session_state.news_df)} 条）")
        else:
            filtered_df = st.session_state.news_df

        if st.button("手动刷新"):
            new_df = get_news()
            if not new_df.empty:
                combined = pd.concat([new_df, st.session_state.news_df]).drop_duplicates(subset=['标题', '发布时间'], keep='first')
                combined = combined.sort_values(by='发布时间', ascending=False)
                st.session_state.news_df = combined.head(MAX_TOTAL)
            st.session_state.last_refresh = time.time()
            st.session_state.last_refresh_str = get_china_time()
            st.rerun()

        # 分页
        total = len(filtered_df)
        total_pages = math.ceil(total / PAGE_SIZE) or 1
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
        page = max(1, min(st.session_state.current_page, total_pages))

        start = (page - 1) * PAGE_SIZE
        page_df = filtered_df.iloc[start:start + PAGE_SIZE]

        # 两列，每列 25 条
        col1, col2 = st.columns(2)

        col1_data = page_df.iloc[0:ITEMS_PER_COLUMN]
        col2_data = page_df.iloc[ITEMS_PER_COLUMN:PAGE_SIZE]

        with col1:
            for _, row in col1_data.iterrows():
                title = row['标题']
                tstr = convert_to_china_time(row['发布时间'])
                if st.button(f"{title}  {tstr}", key=f"left_{title}_{tstr}", use_container_width=True):
                    st.session_state.selected_news = row.to_dict()
                    st.rerun()

        with col2:
            for _, row in col2_data.iterrows():
                title = row['标题']
                tstr = convert_to_china_time(row['发布时间'])
                if st.button(f"{title}  {tstr}", key=f"right_{title}_{tstr}", use_container_width=True):
                    st.session_state.selected_news = row.to_dict()
                    st.rerun()

        # 分页控件（移到外面，确保可见）
        st.markdown("---")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("上一页") and page > 1:
                st.session_state.current_page -= 1
                st.rerun()
        with c2:
            st.caption(f"第 {page} / {total_pages} 页   共 {total} 条（缓存上限 1500 条）")
        with c3:
            if st.button("下一页") and page < total_pages:
                st.session_state.current_page += 1
                st.rerun()

    with col_detail:
        st.subheader("新闻详情 & AI 分析")
        if 'selected_news' in st.session_state:
            news = st.session_state.selected_news
            st.markdown(f"**{news.get('标题')}**")
            st.caption(f"发布时间（中国时间）：{convert_to_china_time(news.get('发布时间', '未知'))}")
            st.info(news.get('内容', '无内容'))
            if news.get('链接'):
                st.markdown(f"[原文链接]({news.get('链接')})")

            if st.button("分析此新闻", type="primary"):
                with st.spinner("分析中..."):
                    try:
                        llm = ChatOpenAI(api_key=API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4/", model="glm-4-flash", temperature=0.3)
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", "你是一位专业的A股研究员。请根据新闻提取1-3个核心概念、3-6只受益个股（带代码+逻辑），Markdown输出。"),
                            ("user", f"标题：{news.get('标题')}\n内容：{news.get('内容')}")
                        ])
                        chain = prompt | llm | StrOutputParser()
                        result = chain.invoke({})
                        st.success("分析完成")
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"分析失败：{str(e)}")
        else:
            st.info("请从左侧选择一条新闻")

        # 手动输入
        st.markdown("---")
        st.subheader("手动输入测试")
        manual_title = st.text_input("标题（可选）")
        manual_content = st.text_area("内容", height=180)
        if st.button("分析手动新闻") and manual_content.strip():
            with st.spinner("分析中..."):
                try:
                    llm = ChatOpenAI(api_key=API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4/", model="glm-4-flash", temperature=0.3)
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "你是一位专业的A股研究员。请根据新闻提取1-3个核心概念、3-6只受益个股（带代码+逻辑），Markdown输出。"),
                        ("user", f"标题：{manual_title}\n内容：{manual_content}")
                    ])
                    chain = prompt | llm | StrOutputParser()
                    result = chain.invoke({})
                    st.success("分析完成")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"分析失败：{str(e)}")

if __name__ == "__main__":
    main()
