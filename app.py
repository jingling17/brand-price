import streamlit as st
import pandas as pd
import numpy as np
import io
import re

# 设置页面配置
st.set_page_config(page_title="电商销售数据分析工具", layout="wide")


# ==========================================
# 1. 核心逻辑函数
# ==========================================

def parse_brand_rules(rules_text):
    """解析品牌合并规则"""
    mapping = {}
    if not rules_text:
        return mapping
    lines = rules_text.strip().split('\n')
    for line in lines:
        if ':' in line or '：' in line:
            parts = line.replace('：', ':').split(':')
            target = parts[0].strip()
            aliases = parts[1].split(',')
            for alias in aliases:
                clean_alias = alias.strip().lower()
                if clean_alias:
                    mapping[clean_alias] = target
    return mapping


def clean_brand_name(name, mapping):
    """清洗品牌名称"""
    s = str(name).lower().strip()
    for alias, target in mapping.items():
        if alias in s:
            return target
    return name


def identify_price_segments(df):
    """
    智能识别价位段列
    逻辑：
    1. 排除 '品牌', 'brand' 等非数值列
    2. 优先选择数值类型的列
    3. 或者列名中包含数字、波浪号、大于小于号的列
    """
    potential_cols = []
    # 排除常见的非价位段列名
    exclude_names = ['品牌', 'brand', 'brands', '序号', 'id', '排名', 'rank']

    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower in exclude_names:
            continue

        # 如果列名包含数字，或者是数值类型，或者是常见的价位段符号
        if (any(char.isdigit() for char in col_lower) or
                '~' in col_lower or '>' in col_lower or '<' in col_lower or
                np.issubdtype(df[col].dtype, np.number)):
            potential_cols.append(col)

    return potential_cols


def load_and_process(file_obj, coeff, mapping):
    """读取并处理单个文件"""
    if file_obj is None:
        return None, None, []

    try:
        if file_obj.name.endswith('.csv'):
            df = pd.read_csv(file_obj)
        else:
            df = pd.read_excel(file_obj)
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return None, None, []

    # --- 1. 自动识别价位段列 ---
    segments = identify_price_segments(df)

    if not segments:
        st.error(f"在文件 {file_obj.name} 中未找到价位段列，请检查表头。")
        return None, None, []

    # --- 2. 数据清洗与系数应用 ---
    for col in segments:
        # 强制转为数字，无法转换的变为NaN然后填0，最后乘系数
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) * coeff

    # --- 3. 严格提取“市场整体”行 ---
    # 为了匹配准确，先转字符串去空格
    # 尝试寻找 '品牌' 列，如果没有，尝试找第一列作为品牌列
    brand_col = '品牌'
    if '品牌' not in df.columns:
        # 简单的回退策略：假设第一列是品牌
        brand_col = df.columns[0]

    df['品牌_clean_temp'] = df[brand_col].astype(str).str.strip()

    market_row = df[df['品牌_clean_temp'] == '市场整体']

    if not market_row.empty:
        market_total = market_row.iloc[0][segments]
    else:
        st.warning(f"⚠️ 文件 `{file_obj.name}` 中未找到 '市场整体' 行！系统将使用累加值代替，数据可能偏小。")
        market_total = df[segments].sum()

    # --- 4. 品牌清洗与汇总 ---
    # 统一使用 '品牌' 作为列名方便后续合并
    if brand_col != '品牌':
        df = df.rename(columns={brand_col: '品牌'})

    df['品牌'] = df['品牌'].apply(lambda x: clean_brand_name(x, mapping))

    # 剔除市场整体
    df_brands = df[df['品牌_clean_temp'] != '市场整体'].copy()

    # 按品牌分组求和
    df_grouped = df_brands.groupby('品牌', as_index=False)[segments].sum()

    # 清理临时列
    if '品牌_clean_temp' in df_grouped.columns:
        del df_grouped['品牌_clean_temp']

    return df_grouped, market_total, segments


def generate_analysis(config, brand_mapping):
    """生成最终报表"""
    # 获取所有年份
    all_years = sorted(list(set([y for p in config.values() for y in p.keys()])))
    if not all_years:
        return None, "未上传任何文件"

    year_data = {}
    valid_years = []
    # 使用集合来收集所有出现过的价位段，保证不漏
    all_seen_segments = []

    # --- 第一轮循环：收集所有可能的价位段并保持顺序 ---
    # 为了保持顺序，我们不能只用 set，得用 list + 查重
    for platform, p_data in config.items():
        for year, item in p_data.items():
            if item['file']:
                # 稍微预读取一下列名（为了效率，这里其实依赖 load_and_process 的结果更稳妥）
                # 所以我们在下面的主循环里动态更新 segments 列表
                pass

    # --- 主数据处理循环 ---
    for year in all_years:
        combined_brands = None
        combined_total = None
        has_data = False

        for platform, p_data in config.items():
            if year in p_data:
                item = p_data[year]
                file_obj = item['file']
                coeff = item['coeff']

                if file_obj:
                    # 指针归零，防止重复读取报错
                    file_obj.seek(0)
                    brands, total, segs = load_and_process(file_obj, coeff, brand_mapping)

                    if brands is not None:
                        has_data = True

                        # 动态更新全局价位段列表（保持顺序）
                        for s in segs:
                            if s not in all_seen_segments:
                                all_seen_segments.append(s)

                        # 对齐数据（如果不同文件价位段不一致，reindex 会补 0）
                        # 这里暂不立即对齐，合并时由 pandas outer join 处理，最后再统一 reindex

                        # 合并逻辑
                        if combined_brands is None:
                            combined_brands = brands
                            combined_total = total  # Series
                        else:
                            # 1. 品牌数据合并
                            combined_brands = pd.concat([combined_brands, brands], ignore_index=True)
                            # 此时列可能增多了，fillna(0) 很重要
                            combined_brands = combined_brands.fillna(0)
                            # 再次 group by
                            # 注意：groupby 时要包括当前所有列
                            cols_to_sum = [c for c in combined_brands.columns if c != '品牌']
                            combined_brands = combined_brands.groupby('品牌', as_index=False)[cols_to_sum].sum()

                            # 2. 市场整体数据合并 (Series add Series，自动对齐索引)
                            combined_total = combined_total.add(total, fill_value=0)

        if has_data:
            valid_years.append(year)
            year_data[year] = {'brands': combined_brands, 'total': combined_total}

    if not valid_years:
        return None, "没有有效数据"

    # --- 生成表格 ---
    # 最终的行索引：所有出现过的价位段
    metrics = pd.DataFrame(index=all_seen_segments)

    for year in valid_years:
        y_total_series = year_data[year]['total']

        # 将 Series 映射到 DataFrame，自动对齐索引，缺失填 0
        metrics[f'{year}销额'] = y_total_series
        metrics[f'{year}销额'] = metrics[f'{year}销额'].fillna(0)

        grand_total = metrics[f'{year}销额'].sum()
        metrics[f'{year}占比'] = metrics[f'{year}销额'] / grand_total if grand_total else 0

    # 计算同比/变化
    if len(valid_years) >= 2:
        y1, y2 = valid_years[-2], valid_years[-1]
        metrics['销额同比'] = (metrics[f'{y2}销额'] - metrics[f'{y1}销额']) / metrics[f'{y1}销额']
        metrics['占比变化'] = metrics[f'{y2}占比'] - metrics[f'{y1}占比']

    # 计算 TOP5
    latest_year = valid_years[-1]
    top5_list = []
    brands_df = year_data[latest_year]['brands']
    total_series = year_data[latest_year]['total']

    for seg in all_seen_segments:
        # 安全获取该价位段总额
        seg_total = total_series.get(seg, 0)

        # 安全获取该价位段品牌排行
        if brands_df is not None and seg in brands_df.columns:
            top = brands_df.sort_values(by=seg, ascending=False).head(5)
            strs = []
            for _, row in top.iterrows():
                brand_sales = row[seg]
                share = brand_sales / seg_total if seg_total > 0 else 0
                strs.append(f"{row['品牌']}({share:.1%})")
            top5_list.append(" ".join(strs))
        else:
            top5_list.append("-")

    metrics[f'{latest_year} TOP5品牌(占比)'] = top5_list

    # 格式化
    def fmt_sales(x):
        try:
            return "{:,.0f}".format(x)
        except:
            return x

    def fmt_pct(x):
        try:
            return "{:.1%}".format(x)
        except:
            return x

    def fmt_change(x):
        try:
            return "{:+.1%}".format(x)
        except:
            return x

    res = metrics.copy()
    for col in res.columns:
        if '销额' in col and '同比' not in col:
            res[col] = res[col].apply(fmt_sales)
        elif '销额同比' in col:
            res[col] = res[col].apply(fmt_pct)
        elif '占比' in col and '变化' not in col and 'TOP5' not in col:
            res[col] = res[col].apply(fmt_pct)
        elif '占比变化' in col:
            res[col] = res[col].apply(fmt_change)

    res = res.reset_index().rename(columns={'index': '价位段'})
    return res, None


# ==========================================
# 2. Streamlit 界面布局
# ==========================================

def main():
    st.title("📊 电商销售数据自动化分析工具")

    # 初始化配置
    config = {'JD': {}, 'Tmall': {}}

    with st.sidebar:
        st.header("1. 品牌合并规则")
        default_rules = """华为: 华为智选, 鸿蒙\npaulmann: paulmann p\n明基: benq, 麦朵尔"""
        rules_input = st.text_area("输入规则", value=default_rules, height=150)
        brand_mapping = parse_brand_rules(rules_input)

        st.header("2. 数据上传")
        st.info("💡 系统将自动识别文件表头中的价位段。")

        # JD 配置
        with st.expander("京东 (JD)", expanded=True):
            jd24_f = st.file_uploader("JD 2024", type=['csv', 'xlsx'], key='jd24')
            jd24_c = st.number_input("JD 24系数", value=0.87, step=0.01, key='c_jd24')
            if jd24_f: config['JD']['2024'] = {'file': jd24_f, 'coeff': jd24_c}

            jd25_f = st.file_uploader("JD 2025", type=['csv', 'xlsx'], key='jd25')
            jd25_c = st.number_input("JD 25系数", value=0.87, step=0.01, key='c_jd25')
            if jd25_f: config['JD']['2025'] = {'file': jd25_f, 'coeff': jd25_c}

        # Tmall 配置
        with st.expander("天猫 (Tmall)", expanded=True):
            tm24_f = st.file_uploader("Tmall 2024", type=['csv', 'xlsx'], key='tm24')
            tm24_c = st.number_input("Tmall 24系数", value=0.82, step=0.01, key='c_tm24')
            if tm24_f: config['Tmall']['2024'] = {'file': tm24_f, 'coeff': tm24_c}

            tm25_f = st.file_uploader("Tmall 2025", type=['csv', 'xlsx'], key='tm25')
            tm25_c = st.number_input("Tmall 25系数", value=0.72, step=0.01, key='c_tm25')
            if tm25_f: config['Tmall']['2025'] = {'file': tm25_f, 'coeff': tm25_c}

    # --- 自动检测是否运行 ---
    has_file = any(config['JD']) or any(config['Tmall'])

    if not has_file:
        st.info("👈 请在左侧侧边栏上传 Excel/CSV 文件，系统将自动开始分析。")
        return

    st.divider()

    # 1. 总体合并表
    df_combined, err = generate_analysis(config, brand_mapping)

    if df_combined is not None:
        st.subheader("🏆 JD + Tmall 渠道汇总")
        st.dataframe(df_combined, use_container_width=True)
        csv = df_combined.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载汇总表 (CSV)", csv, "combined_analysis.csv", "text/csv", type='primary')
    elif err:
        st.error(err)

    # 2. 分平台表
    col1, col2 = st.columns(2)

    with col1:
        if any(config['JD']):
            st.subheader("🐶 京东 (JD)")
            jd_conf = {'JD': config['JD']}
            df_jd, _ = generate_analysis(jd_conf, brand_mapping)
            st.dataframe(df_jd, use_container_width=True)

    with col2:
        if any(config['Tmall']):
            st.subheader("🐱 天猫 (Tmall)")
            tm_conf = {'Tmall': config['Tmall']}
            df_tm, _ = generate_analysis(tm_conf, brand_mapping)
            st.dataframe(df_tm, use_container_width=True)


if __name__ == "__main__":
    main()