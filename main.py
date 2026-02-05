import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import akshare as ak
import warnings
import time
import urllib3
import pytz
import requests
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

st.markdown("""
    <style>
    .stButton > button {
        font-size: 13px !important;
        padding: 6px 10px !important;
        line-height: 1.1 !important;
        min-height: 55px !important;
        margin-bottom: 3px !important;
        white-space: normal !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        width: 100% !important;
    }
    .stColumns > div {
        padding: 0 5px !important;
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
    except Exception as e:
        st.error(f"抓取失败: {str(e)}")
        return pd.DataFrame(columns=['标题', '内容', '发布时间', '链接'])

def get_china_time():
    china_tz = pytz.timezone('Asia/Shanghai')
    return datetime.now(china_tz).strftime("%Y-%m-%d %H:%M:%S")

def convert_to_china_time(time_str):
    if time_str in ['未知', '未知时间', None]:
        return time_str
    try:
        pub_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        china_tz = pytz.timezone('Asia/Shanghai')
        return pub_time.replace(tzinfo=pytz.UTC).astimezone(china_tz).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return time_str

def main():
    st.title("AI 新闻概念 & 个股挖掘工具")
    st.caption("点击标题查看详情和AI分析")

    if 'news_df' not in st.session_state:
        st.session_state.news_df = get_news()
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    if 'last_refresh_str' not in st.session_state:
        st.session_state.last_refresh_str = get_china_time()

    current_time = time.time()
    if current_time - st.session_state.last_refresh > REFRESH_INTERVAL:
        new_df = get_news()
        if not new_df.empty:
            combined = pd.concat([new_df, st.session_state.news_df])
            combined = combined.drop_duplicates(subset=['标题', '发布时间'], keep='first')
            combined = combined.sort_values(by='发布时间', ascending=False)
            st.session_state.news_df = combined.head(MAX_TOTAL)
        st.session_state.last_refresh = current_time
        st.session_state.last_refresh_str = get_china_time()
        st.rerun()

    col_list, col_detail = st.columns([7, 3])

    with col_list:
        st.subheader("最新财经快讯")
        st.caption(f"上次刷新: {st.session_state.last_refresh_str}（每2分钟自动）")

        search_keyword = st.text_input("搜索（支持全缓存搜索）", "")
        search_keyword = search_keyword.strip().lower()

        if search_keyword and ('prev_search' not in st.session_state or st.session_state.prev_search != search_keyword):
            st.session_state.current_page = 1
            st.session_state.prev_search = search_keyword

        if search_keyword:
            filtered_df = st.session_state.news_df[
                st.session_state.news_df['标题'].str.lower().str.contains(search_keyword, na=False) |
                st.session_state.news_df['内容'].str.lower().str.contains(search_keyword, na=False)
            ]
            st.info(f"找到 {len(filtered_df)} 条（缓存 {len(st.session_state.news_df)} 条）")
        else:
            filtered_df = st.session_state.news_df

        if st.button("手动刷新新闻列表"):
            new_df = get_news()
            if not new_df.empty:
                combined = pd.concat([new_df, st.session_state.news_df])
                combined = combined.drop_duplicates(subset=['标题', '发布时间'], keep='first')
                combined = combined.sort_values(by='发布时间', ascending=False)
                st.session_state.news_df = combined.head(MAX_TOTAL)
            st.session_state.last_refresh = time.time()
            st.session_state.last_refresh_str = get_china_time()
            st.rerun()

        total = len(filtered_df)
        total_pages = math.ceil(total / PAGE_SIZE) or 1
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
        page = max(1, min(st.session_state.current_page, total_pages))

        start = (page - 1) * PAGE_SIZE
        page_df = filtered_df.iloc[start:start + PAGE_SIZE]

        col1, col2 = st.columns(2)

        col1_data = page_df.iloc[0:ITEMS_PER_COLUMN]
        col2_data = page_df.iloc[ITEMS_PER_COLUMN:PAGE_SIZE]

        with col1:
            for _, row in col1_data.iterrows():
                title = row['标题']
                tstr = convert_to_china_time(row['发布时间'])
                btn_key = f"btn_left_{title}_{tstr}"
                if st.button(f"{title}  {tstr}", key=btn_key, use_container_width=True):
                    st.session_state.selected_news = row.to_dict()
                    st.rerun()

        with col2:
            for _, row in col2_data.iterrows():
                title = row['标题']
                tstr = convert_to_china_time(row['发布时间'])
                btn_key = f"btn_right_{title}_{tstr}"
                if st.button(f"{title}  {tstr}", key=btn_key, use_container_width=True):
                    st.session_state.selected_news = row.to_dict()
                    st.rerun()

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

            if st.button("用 GLM-4-Flash 分析", type="primary"):
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


