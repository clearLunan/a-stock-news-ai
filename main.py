import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import akshare as ak
import warnings
import requests
import time
import os
import urllib3
import pytz  # 用于中国时区
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
requests.packages.urllib3.disable_warnings()

# ================= 配置 =================
st.set_page_config(
    page_title="AI 财经新闻概念挖掘机（同花顺版）",
    page_icon="📈",
    layout="wide"
)

API_KEY = os.getenv("ZHIPU_API_KEY")  # 从环境变量读取（Streamlit Cloud Secrets）

REFRESH_INTERVAL = 120  # 2分钟 = 120秒

def get_news():
    try:
        df = ak.stock_info_global_ths()
        required_cols = ['标题', '内容', '发布时间', '链接']
        for col in required_cols:
            if col not in df.columns:
                df[col] = '未知'
        df = df.head(50)
        return df
    except Exception as e:
        st.error(f"抓取同花顺新闻失败: {str(e)}\n（可能网络/SSL问题，试手动输入测试。）")
        return pd.DataFrame(columns=['标题', '内容', '发布时间', '链接'])

def get_china_time():
    """获取当前中国时间字符串"""
    china_tz = pytz.timezone('Asia/Shanghai')
    return datetime.now(china_tz).strftime("%Y-%m-%d %H:%M:%S")

def main():
    st.title("AI 新闻概念 & 个股挖掘工具（同花顺版）")
    st.caption("点击左侧新闻标题 → 右侧显示详情 → 点击分析按钮获取 AI 解读")

    # 初始化 session_state
    if 'news_df' not in st.session_state:
        st.session_state.news_df = get_news()
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    if 'last_refresh_str' not in st.session_state:
        st.session_state.last_refresh_str = get_china_time()

    # 自动刷新检查
    current_time = time.time()
    if current_time - st.session_state.last_refresh > REFRESH_INTERVAL:
        st.session_state.news_df = get_news()
        st.session_state.last_refresh = current_time
        st.session_state.last_refresh_str = get_china_time()
        st.rerun()

    col_list, col_detail = st.columns([3, 7])

    with col_list:
        st.subheader("最新财经快讯（同花顺）")
        st.caption(f"上次刷新: {st.session_state.last_refresh_str}（自动每2分钟检查一次）")

        search_keyword = st.text_input("搜索新闻（关键词）", placeholder="输入标题或内容关键词...", key="search")
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
            st.session_state.news_df = get_news()
            st.session_state.last_refresh = time.time()
            st.session_state.last_refresh_str = get_china_time()
            st.rerun()

        if not filtered_df.empty:
            for idx, row in filtered_df.iterrows():
                title = row.get('标题', '无标题')
                # 新闻发布时间转中国时间（假设原始是 UTC 或无时区）
                pub_time_str = row.get('发布时间', '未知时间')
                if pub_time_str != '未知时间':
                    try:
                        # 如果原始时间字符串有格式问题，可调整 strptime 格式
                        pub_time = datetime.strptime(pub_time_str, "%Y-%m-%d %H:%M:%S")
                        china_tz = pytz.timezone('Asia/Shanghai')
                        pub_time_china = pub_time.replace(tzinfo=pytz.UTC).astimezone(china_tz) if 'UTC' in pub_time_str else pub_time.replace(tzinfo=china_tz)
                        time_str = pub_time_china.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        time_str = pub_time_str  # 如果转换失败，原样显示
                else:
                    time_str = '未知时间'
                btn_text = f"{title}\n{time_str}"
                if st.button(btn_text, key=f"news_btn_{idx}", use_container_width=True):
                    st.session_state.selected_idx = idx
                    st.rerun()
        else:
            st.info("暂无匹配新闻或加载失败，用下方手动输入测试。")

    with col_detail:
        if 'selected_idx' in st.session_state:
            idx = st.session_state.selected_idx
            if idx < len(st.session_state.news_df):
                news = st.session_state.news_df.iloc[idx]
                st.subheader(news.get('标题', '标题加载中'))
                # 发布时间转中国时间
                pub_time_str = news.get('发布时间', '未知')
                if pub_time_str != '未知':
                    try:
                        pub_time = datetime.strptime(pub_time_str, "%Y-%m-%d %H:%M:%S")
                        china_tz = pytz.timezone('Asia/Shanghai')
                        pub_time_china = pub_time.replace(tzinfo=pytz.UTC).astimezone(china_tz) if 'UTC' in pub_time_str else pub_time.replace(tzinfo=china_tz)
                        st.caption(f"发布时间（中国时间）：{pub_time_china.strftime('%Y-%m-%d %H:%M:%S')}")
                    except:
                        st.caption(f"发布时间：{pub_time_str}")
                else:
                    st.caption("发布时间：未知")
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

        # 手动输入备用
        st.markdown("---")
        st.subheader("手动输入新闻测试（备用）")
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

        # 自动刷新设置（只用 JS 方案）
        st.markdown("---")
        st.subheader("自动刷新设置")

        # 显示倒计时（中国时间）
        if 'last_refresh' in st.session_state:
            elapsed = time.time() - st.session_state.last_refresh
            remaining = max(0, REFRESH_INTERVAL - elapsed)
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            st.caption(f"下次建议刷新剩余：{minutes}分 {seconds}秒（点击下方按钮开启自动）")

        if st.button("开启自动页面刷新（每2分钟自动重载一次）"):
            st.success("已开启！浏览器将每2分钟自动刷新页面，保持新闻最新。")
            auto_js = f"""
            <script>
                function autoReload() {{
                    window.location.reload(true);
                }}
                setInterval(autoReload, {REFRESH_INTERVAL * 1000});
            </script>
            """
            st.components.v1.html(auto_js, height=0)

        if st.button("立即手动刷新新闻列表"):
            st.session_state.news_df = get_news()
            st.session_state.last_refresh = time.time()
            st.session_state.last_refresh_str = get_china_time()
            st.rerun()

if __name__ == "__main__":
    main()

