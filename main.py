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

# 忽略无关警告，避免页面显示干扰
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
requests.packages.urllib3.disable_warnings()

# 页面基础配置：标题、图标、宽布局
st.set_page_config(page_title="AI 财经新闻概念挖掘机", page_icon="📈", layout="wide")

# ========== 核心配置（关键修复点） ==========
API_KEY = os.getenv("ZHIPU_API_KEY")          # 智谱API密钥（环境变量配置）
REFRESH_INTERVAL = 120                       # 自动刷新间隔（秒）
PAGE_SIZE = 50                               # 单页总条数（2列×25条）
ITEMS_PER_COLUMN = 25                        # 每列固定25条
MAX_TOTAL = 1500                             # 缓存最大条数（匹配30页）
MAX_PAGES = 30                               # 最大分页页数（强制限制30页）

# ========== 样式优化（确保两列正常显示，按钮排版美观） ==========
st.markdown("""
    <style>
    /* 按钮样式：统一大小、换行、宽度100% */
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
    /* 强制两列均分，避免挤压成一列 */
    .col-container {
        display: flex;
        gap: 10px;
        width: 100%;
    }
    .col-item {
        flex: 1;
        min-width: 0;
    }
    /* 分页按钮居中对齐 */
    .stHorizontalBlock {
        align-items: center !important;
        justify-content: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

def get_news():
    """抓取财经新闻，扩大抓取量，确保分页有足够数据"""
    try:
        # 从akshare获取最新财经新闻
        df = ak.stock_info_global_ths()
        # 确保关键列存在，避免报错
        required_cols = ['标题', '内容', '发布时间', '链接']
        for col in required_cols:
            if col not in df.columns:
                df[col] = '未知'
        # 抓取最多MAX_TOTAL条数据，保证30页的数据源
        return df.head(MAX_TOTAL)
    except Exception as e:
        st.error(f"新闻抓取失败: {str(e)}")
        return pd.DataFrame(columns=['标题', '内容', '发布时间', '链接'])

def get_china_time():
    """获取当前中国时区（北京时间）"""
    china_tz = pytz.timezone('Asia/Shanghai')
    return datetime.now(china_tz).strftime("%Y-%m-%d %H:%M:%S")

def convert_to_china_time(time_str):
    """简化时间转换：akshare返回的已是北京时间，仅做空值处理"""
    if time_str in ['未知', '未知时间', None]:
        return time_str
    return time_str

def main():
    # 页面标题和说明
    st.title("AI 新闻概念 & 个股挖掘工具")
    st.caption("点击标题查看详情和AI分析 | 每2分钟自动刷新新闻")

    # ========== 会话状态初始化（避免页面刷新丢失数据） ==========
    if 'news_df' not in st.session_state:
        st.session_state.news_df = get_news()                # 新闻数据缓存
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()          # 最后刷新时间戳
    if 'last_refresh_str' not in st.session_state:
        st.session_state.last_refresh_str = get_china_time() # 最后刷新时间（格式化）
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1                    # 当前页码
    if 'prev_search' not in st.session_state:
        st.session_state.prev_search = ""                    # 上一次搜索关键词

    # ========== 自动刷新逻辑（原生autorefresh，无页面闪烁） ==========
    st.autorefresh(interval=REFRESH_INTERVAL * 1000, key="auto-refresh")
    current_time = time.time()
    if current_time - st.session_state.last_refresh > REFRESH_INTERVAL:
        new_df = get_news()
        if not new_df.empty:
            # 合并新数据+旧缓存，去重，按时间倒序，限制总条数
            combined = pd.concat([new_df, st.session_state.news_df])
            combined = combined.drop_duplicates(subset=['标题', '发布时间'], keep='first')
            combined = combined.sort_values(by='发布时间', ascending=False)
            st.session_state.news_df = combined.head(MAX_TOTAL)
        # 更新刷新时间
        st.session_state.last_refresh = current_time
        st.session_state.last_refresh_str = get_china_time()

    # ========== 主布局：左侧新闻列表（7份）+ 右侧详情/分析（3份） ==========
    col_list, col_detail = st.columns([7, 3])

    with col_list:
        st.subheader("最新财经快讯")
        st.caption(f"上次刷新: {st.session_state.last_refresh_str}（手动刷新点击下方按钮）")

        # ========== 搜索功能 ==========
        search_keyword = st.text_input("搜索（支持标题/内容关键词）", "")
        search_keyword = search_keyword.strip().lower()

        # 搜索关键词变化时，重置页码为1
        if search_keyword and st.session_state.prev_search != search_keyword:
            st.session_state.current_page = 1
            st.session_state.prev_search = search_keyword

        # 根据关键词过滤数据
        if search_keyword:
            filtered_df = st.session_state.news_df[
                st.session_state.news_df['标题'].str.lower().str.contains(search_keyword, na=False) |
                st.session_state.news_df['内容'].str.lower().str.contains(search_keyword, na=False)
            ]
            # 搜索结果也限制最大1500条（30页）
            filtered_df = filtered_df.head(MAX_TOTAL)
            st.info(f"搜索结果：{len(filtered_df)} 条（缓存总数：{len(st.session_state.news_df)} 条）")
        else:
            filtered_df = st.session_state.news_df

        # ========== 手动刷新按钮 ==========
        if st.button("手动刷新新闻列表", use_container_width=True):
            new_df = get_news()
            if not new_df.empty:
                combined = pd.concat([new_df, st.session_state.news_df])
                combined = combined.drop_duplicates(subset=['标题', '发布时间'], keep='first')
                combined = combined.sort_values(by='发布时间', ascending=False)
                st.session_state.news_df = combined.head(MAX_TOTAL)
            st.session_state.last_refresh = time.time()
            st.session_state.last_refresh_str = get_china_time()

        # ========== 分页逻辑（核心修复：限制最大30页） ==========
        total = len(filtered_df)
        # 计算总页数：总条数/单页条数，且不超过30页
        total_pages = min(math.ceil(total / PAGE_SIZE) or 1, MAX_PAGES)
        # 确保当前页在1~30页之间
        current_page = max(1, min(st.session_state.current_page, total_pages))

        # 截取当前页数据（单页50条）
        start_idx = (current_page - 1) * PAGE_SIZE
        page_df = filtered_df.iloc[start_idx:start_idx + PAGE_SIZE]

        # ========== 两列展示（核心修复：每列固定25条） ==========
        st.markdown('<div class="col-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)  # 强制均分两列

        # 左列：前25条
        with col1:
            st.markdown('<div class="col-item">', unsafe_allow_html=True)
            col1_data = page_df.iloc[0:ITEMS_PER_COLUMN]
            for _, row in col1_data.iterrows():
                title = row['标题']
                pub_time = convert_to_china_time(row['发布时间'])
                # 按钮唯一key，避免重复报错
                btn_key = f"news_btn_{current_page}_left_{_}"
                if st.button(f"{title}\n{pub_time}", key=btn_key, use_container_width=True):
                    st.session_state.selected_news = row.to_dict()
            st.markdown('</div>', unsafe_allow_html=True)

        # 右列：后25条
        with col2:
            st.markdown('<div class="col-item">', unsafe_allow_html=True)
            col2_data = page_df.iloc[ITEMS_PER_COLUMN:PAGE_SIZE]
            for _, row in col2_data.iterrows():
                title = row['标题']
                pub_time = convert_to_china_time(row['发布时间'])
                btn_key = f"news_btn_{current_page}_right_{_}"
                if st.button(f"{title}\n{pub_time}", key=btn_key, use_container_width=True):
                    st.session_state.selected_news = row.to_dict()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ========== 分页按钮 ==========
        st.markdown("---")
        btn_prev, page_info, btn_next = st.columns([1, 2, 1])
        with btn_prev:
            if st.button("上一页", use_container_width=True) and current_page > 1:
                st.session_state.current_page -= 1
        with page_info:
            st.caption(f"第 {current_page} / {total_pages} 页 | 总计 {total} 条 | 最大30页")
        with btn_next:
            if st.button("下一页", use_container_width=True) and current_page < total_pages:
                st.session_state.current_page += 1

    with col_detail:
        st.subheader("新闻详情 & AI 分析")
        # 显示选中的新闻详情
        if 'selected_news' in st.session_state:
            news = st.session_state.selected_news
            st.markdown(f"### {news.get('标题')}")
            st.caption(f"发布时间：{convert_to_china_time(news.get('发布时间', '未知'))}")
            st.divider()
            st.info(f"内容：{news.get('内容', '无内容')}")
            if news.get('链接') and news.get('链接') != '未知':
                st.markdown(f"[原文链接]({news.get('链接')})", unsafe_allow_html=True)

            # ========== AI分析功能（智谱GLM-4-Flash） ==========
            if st.button("用 GLM-4-Flash 分析", type="primary", use_container_width=True):
                with st.spinner("AI正在分析，请稍候..."):
                    try:
                        # 检查API密钥是否配置
                        if not API_KEY:
                            st.error("❌ 未配置智谱API密钥！\n请在服务器执行：export ZHIPU_API_KEY='你的密钥'")
                        else:
                            # 初始化LLM模型
                            llm = ChatOpenAI(
                                api_key=API_KEY,
                                base_url="https://open.bigmodel.cn/api/paas/v4/",
                                model="glm-4-flash",
                                temperature=0.3  # 低随机性，保证分析准确
                            )
                            # 分析提示词
                            prompt = ChatPromptTemplate.from_messages([
                                ("system", """你是专业的A股财经研究员，需完成以下分析：
1. 从新闻中提取1-3个核心概念（如：人工智能、新能源、半导体）；
2. 推荐3-6只受益个股（需包含股票代码+受益逻辑）；
3. 所有内容用Markdown格式输出，条理清晰。"""),
                                ("user", f"新闻标题：{news.get('标题')}\n新闻内容：{news.get('内容')}")
                            ])
                            # 执行分析
                            chain = prompt | llm | StrOutputParser()
                            analysis_result = chain.invoke({})
                            st.success("✅ AI分析完成！")
                            st.markdown(analysis_result)
                    except Exception as e:
                        st.error(f"❌ 分析失败：{str(e)}")
        else:
            st.info("请从左侧新闻列表中点击一条新闻，查看详情和AI分析")

        # ========== 手动输入测试功能 ==========
        st.divider()
        st.subheader("手动输入新闻分析")
        manual_title = st.text_input("新闻标题（可选）")
        manual_content = st.text_area("新闻内容", height=180, placeholder="请输入需要分析的新闻内容...")
        if st.button("分析手动输入内容", use_container_width=True) and manual_content.strip():
            with st.spinner("AI正在分析，请稍候..."):
                try:
                    if not API_KEY:
                        st.error("❌ 未配置智谱API密钥！\n请在服务器执行：export ZHIPU_API_KEY='你的密钥'")
                    else:
                        llm = ChatOpenAI(
                            api_key=API_KEY,
                            base_url="https://open.bigmodel.cn/api/paas/v4/",
                            model="glm-4-flash",
                            temperature=0.3
                        )
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", """你是专业的A股财经研究员，需完成以下分析：
1. 从新闻中提取1-3个核心概念（如：人工智能、新能源、半导体）；
2. 推荐3-6只受益个股（需包含股票代码+受益逻辑）；
3. 所有内容用Markdown格式输出，条理清晰。"""),
                            ("user", f"新闻标题：{manual_title}\n新闻内容：{manual_content}")
                        ])
                        chain = prompt | llm | StrOutputParser()
                        result = chain.invoke({})
                        st.success("✅ AI分析完成！")
                        st.markdown(result)
                except Exception as e:
                    st.error(f"❌ 分析失败：{str(e)}")

if __name__ == "__main__":
    main()
