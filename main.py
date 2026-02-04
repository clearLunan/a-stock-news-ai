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
PAGE_SIZE = 50          # 每页 50 条（两列 × 25）
ITEMS_PER_COLUMN = 25   # 每列 25 条
MAX_PAGES = 30
MAX_TOTAL = MAX_PAGES * PAGE_SIZE

# 小字体 CSS
st.markdown("""
    <style>
    .news-button {
        font-size: 14px !important;
        padding: 8px 12px !important;
        line-height: 1.2 !important;
        height: auto !important;
        white-space: normal !important;
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
    st.caption("点击左侧新闻标题 → 右侧显示详情及 AI 分析")

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
            combined = pd.concat([new_df, st.session_state.news_df]).drop_duplicates(subset=['标题', '发布时间'], keep='first')
            combined = combined.sort_values(by='发布时间', ascending=False)
            st.session_state.news_df = combined.head(MAX_TOTAL)
        st.session_state.last_refresh = current_time
        st.session_state.last_refresh_str = get_china_time()
        st.rerun()

    # 列宽：左侧大（新闻列表），右侧小（详情）
    col_list, col_detail = st.columns([7, 3])

    with col_list:
        st.subheader("最新财经快讯（同花顺）")
        st.caption(f"上次刷新: {st.session_state.last_refresh_str}（自动每2分钟检查一次）")

        search_keyword = st.text_input("搜索新闻（关键词）", placeholder="输入标题或内容关键词，可搜索所有缓存内容...")
        search_keyword = search_keyword.strip().lower()

        if search_keyword:
            filtered_df = st.session_state.news_df[
                st.session_state.news_df['标题'].str.lower().str.contains(search_keyword, na=False) |
                st.session_state.news_df['内容'].str.lower().str.contains(search_keyword, na=False)
            ]
            st.info(f"搜索到 {len(filtered_df)} 条匹配新闻")
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

        st.session_state.current_page = max(1, min(st.session_state.current_page, total_pages))

        start_idx = (st.session_state.current_page - 1) * PAGE_SIZE
        end_idx = start_idx + PAGE_SIZE
        page_df = filtered_df.iloc[start_idx:end_idx]

        # 两列显示新闻
        col_news1, col_news2 = st.columns(2)
        page_items = page_df.to_dict('records')

        # 分成两列
        left_items = page_items[:ITEMS_PER_COLUMN]
        right_items = page_items[ITEMS_PER_COLUMN:]

        with col_news1:
            for item in left_items:
                title = item.get('标题', '无标题')
                time_str = convert_to_china_time(item.get('发布时间', '未知时间'))
                btn_text = f"{title}   {time_str}"
                if st.button(btn_text, key=f"news_btn_{item.get('标题')}_{time_str}", use_container_width=True, help="点击查看详情并分析"):
                    st.session_state.selected_idx = page_df.index[page_df['标题'] == title].tolist()[0]
                    st.rerun()

        with col_news2:
            for item in right_items:
                title = item.get('标题', '无标题')
                time_str = convert_to_china_time(item.get('发布时间', '未知时间'))
                btn_text = f"{title}   {time_str}"
                if st.button(btn_text, key=f"news_btn_{item.get('标题')}_{time_str}", use_container_width=True, help="点击查看详情并分析"):
                    st.session_state.selected_idx = page_df.index[page_df['标题'] == title].tolist()[0]
                    st.rerun()

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

    with col_detail:
        st.subheader("新闻详情 & AI 分析")
        if 'selected_idx' in st.session_state:
            idx = st.session_state.selected_idx
            if idx < len(st.session_state.news_df):
                news = st.session_state.news_df.iloc[idx]
                st.markdown(f"**标题：** {news.get('标题', '无标题')}")
                pub_time_str = news.get('发布时间', '未知')
                china_pub_time = convert_to_china_time(pub_time_str)
                st.caption(f"发布时间（中国时间）：{china_pub_time}")
                st.info(news.get('内容', '内容暂不可见'))
                if news.get('链接'):
                    st.markdown(f"[原文链接]({news.get('链接')})")

                if st.button("用 GLM-4-Flash 分析概念 & 个股", type="primary", use_container_width=True):
                    with st.spinner("AI 正在分析...（约5-15秒）"):
                        try:
                            llm = ChatOpenAI(
                                api_key=API_KEY,
                                base_url="https://open.bigmodel.cn/api/paas/v4/",
                                model="glm-4-flash",
                                temperature=0.3
                            )
                            prompt = ChatPromptTemplate.from_messages([
                                ("system", """你是一位专业的A股/港股研究员。
请严格根据下面新闻内容分析：
1. 提取 1-3 个最核心的炒作概念（强势、热点优先）
2. 列出 3-6 只最可能短期受益的个股（带代码，优先A股龙头）
3. 每只个股简述 1-2 句受益逻辑
输出使用 Markdown 格式，结构清晰。"""),
                                ("user", "标题：{title}\n内容：{content}\n请开始分析。")
                            ])
                            chain = prompt | llm | StrOutputParser()
                            result = chain.invoke({
                                "title": news.get('标题', ''),
                                "content": news.get('内容', '')
                            })
                            st.success("分析完成！")
                            st.markdown(result)
                        except Exception as e:
                            st.error(f"AI 分析失败：{str(e)}")
            else:
                st.info("请选择左侧新闻查看详情")
        else:
            st.info("点击左侧新闻查看详情和 AI 分析")

        # 手动输入
        st.markdown("---")
        st.subheader("手动输入新闻测试")
        manual_title = st.text_input("新闻标题（可选）")
        manual_content = st.text_area("新闻内容（粘贴全文）", height=200)
        if st.button("分析这条手动新闻") and manual_content.strip():
            with st.spinner("分析中..."):
                try:
                    llm = ChatOpenAI(
                        api_key=API_KEY,
                        base_url="https://open.bigmodel.cn/api/paas/v4/",
                        model="glm-4-flash",
                        temperature=0.3
                    )
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", """你是一位专业的A股研究员。请严格根据新闻分析：
1. 提取1-3个核心炒作概念（强势优先）
2. 列出3-6只最可能受益个股（带代码，优先龙头）
3. 每只个股1-2句受益逻辑
Markdown 格式输出。"""),
                        ("user", f"标题：{manual_title}\n内容：{manual_content}\n开始分析。")
                    ])
                    chain = prompt | llm | StrOutputParser()
                    result = chain.invoke({})
                    st.success("手动分析完成！")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"手动分析失败：{str(e)}")

        # 自动刷新
        st.markdown("---")
        st.subheader("自动刷新设置")
        if 'last_refresh' in st.session_state:
            elapsed = time.time() - st.session_state.last_refresh
            remaining = max(0, REFRESH_INTERVAL - elapsed)
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            st.caption(f"下次建议刷新剩余：{minutes}分 {seconds}秒")

        if st.button("开启自动页面刷新（每2分钟自动重载一次）"):
            st.success("已开启！浏览器将每2分钟自动刷新页面。")
            auto_js = f"""
            <script>
                function autoReload() {{
                    window.location.reload(true);
                }}
                setInterval(autoReload, {REFRESH_INTERVAL * 1000});
            </script>
            """
            st.components.v1.html(auto_js, height=0)

if __name__ == "__main__":
    main()
