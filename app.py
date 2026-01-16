import streamlit as st
import pandas as pd
import numpy as np
import plotly
import io


# 设置页面配置
st.set_page_config(page_title="电商销售数据分析工具", layout="wide")


# ==========================================
# 1. 核心逻辑函数
# ==========================================

def parse_brand_rules(rules_text):
    """
    解析用户输入的品牌合并规则
    格式: 目标品牌: 别名1, 别名2
    """
    mapping = {}
    if not rules_text:
        return mapping

    lines = rules_text.strip().split('\n')
    for line in lines:
        if ':' in line or '：' in line:
            # 兼容中英文冒号
            parts = line.replace('：', ':').split(':')
            target = parts[0].strip()
            aliases = parts[1].split(',')
            # 将每个别名映射到目标品牌
            for alias in aliases:
                clean_alias = alias.strip().lower()
                if clean_alias:
                    mapping[clean_alias] = target
    return mapping


def clean_brand_name(name, mapping):
    """根据映射表清洗品牌"""
    s = str(name).lower().strip()

    # 1. 优先匹配用户自定义规则 (模糊匹配)
    for alias, target in mapping.items():
        if alias in s:
            return target

    # 2. 返回原名
    return name


def load_and_process(file_obj, coeff, mapping):
    """
    读取并处理单个文件
    重点：严格提取'市场整体'行作为总数据
    """
    if file_obj is None:
        return None, None, None

    try:
        if file_obj.name.endswith('.csv'):
            df = pd.read_csv(file_obj)
        else:
            df = pd.read_excel(file_obj)
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return None, None, None

    # 1. 自动识别数值列（价位段）
    # 排除 '品牌' 列，其他一般为价位段
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 如果有非数字列混入（比如文本格式的价位段），尝试转换
    segments = []
    for col in df.columns:
        if col != '品牌' and col != 'Brand':  # 简单排除
            segments.append(col)

    # 2. 数据清洗与系数应用
    for col in segments:
        # 强制转为数字，无法转换的变为NaN然后填0，最后乘系数
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) * coeff

    # 3. 【关键修改】严格提取“市场整体”行
    # 先将品牌列转为字符串并去空格，防止 "市场整体 " 匹配失败
    df['品牌_clean'] = df['品牌'].astype(str).str.strip()

    market_row = df[df['品牌_clean'] == '市场整体']

    if not market_row.empty:
        # 取第一条匹配到的（通常只有一条），提取所有价位段的数据 Series
        market_total = market_row.iloc[0][segments]
    else:
        # 如果实在找不到，给予警告，并无奈使用求和（此时数据可能会偏小）
        st.warning(
            f"⚠️ 文件 `{file_obj.name}` 中未找到 '市场整体' 行！系统将使用品牌累加值代替，数据可能偏小。请检查源文件品牌列是否有 '市场整体'。")
        market_total = df[segments].sum()

    # 4. 品牌清洗与汇总（用于TOP5）
    # 清洗品牌名称
    df['品牌'] = df['品牌'].apply(lambda x: clean_brand_name(x, mapping))

    # 剔除市场整体，并按清洗后的品牌名分组求和
    df_brands = df[df['品牌_clean'] != '市场整体'].copy()
    df_grouped = df_brands.groupby('品牌', as_index=False)[segments].sum()

    # 清理临时列
    if '品牌_clean' in df_grouped.columns:
        del df_grouped['品牌_clean']

    return df_grouped, market_total, segments


def generate_analysis(config, brand_mapping):
    """生成最终报表"""
    # 获取所有年份
    all_years = sorted(list(set([y for p in config.values() for y in p.keys()])))
    if not all_years:
        return None, "未上传任何文件"

    # 存储处理后的数据
    year_data = {}
    valid_years = []
    final_segments = []

    for year in all_years:
        combined_brands = None
        combined_total = None  # 这是一个Series，索引是价位段
        has_data = False

        for platform, p_data in config.items():
            if year in p_data:
                item = p_data[year]
                file_obj = item['file']
                coeff = item['coeff']

                if file_obj:
                    brands, total, segs = load_and_process(file_obj, coeff, brand_mapping)
                    if brands is not None:
                        has_data = True
                        final_segments = segs  # 更新价位段列表

                        # 合并逻辑：品牌数据合并
                        if combined_brands is None:
                            combined_brands = brands
                            combined_total = total
                        else:
                            combined_brands = pd.concat([combined_brands, brands], ignore_index=True)
                            combined_brands = combined_brands.groupby('品牌', as_index=False)[segs].sum()
                            # 【关键】总数据直接累加（JD市场整体 + Tmall市场整体）
                            combined_total = combined_total.add(total, fill_value=0)

        if has_data:
            valid_years.append(year)
            year_data[year] = {'brands': combined_brands, 'total': combined_total}

    if not valid_years:
        return None, "没有有效数据"

    # --- 生成表格 ---
    # 行索引为价位段
    metrics = pd.DataFrame(index=final_segments)

    for year in valid_years:
        y_total_series = year_data[year]['total']  # 这是该年份各价位段的“市场整体”之和

        # 填充到表中
        # 注意：这里要确保 Series 的索引和 metrics 的索引对齐
        # 如果文件列顺序不一致可能会有问题，这里假设一致
        metrics[f'{year}销额'] = y_total_series

        # 计算该年的总盘子（所有价位段之和）
        grand_total = y_total_series.sum()
        metrics[f'{year}占比'] = metrics[f'{year}销额'] / grand_total if grand_total else 0

    # 计算同比/变化 (如果有2年以上数据，取最后两年)
    if len(valid_years) >= 2:
        y1, y2 = valid_years[-2], valid_years[-1]
        # 销额同比 (增长率)
        metrics['销额同比'] = (metrics[f'{y2}销额'] - metrics[f'{y1}销额']) / metrics[f'{y1}销额']
        # 占比变化 (百分点差值)
        metrics['占比变化'] = metrics[f'{y2}占比'] - metrics[f'{y1}占比']

    # 计算 TOP5 (取最后一年)
    latest_year = valid_years[-1]
    top5_list = []
    brands_df = year_data[latest_year]['brands']
    total_series = year_data[latest_year]['total']

    for seg in final_segments:
        # 该价位段的市场总额（来自市场整体行）
        seg_total = total_series[seg] if seg in total_series else 0

        # 对该价位段品牌排序
        if seg in brands_df.columns:
            top = brands_df.sort_values(by=seg, ascending=False).head(5)
            strs = []
            for _, row in top.iterrows():
                brand_sales = row[seg]
                # 占比 = 品牌销额 / 市场整体销额
                share = brand_sales / seg_total if seg_total > 0 else 0
                strs.append(f"{row['品牌']}({share:.1%})")
            top5_list.append(" ".join(strs))
        else:
            top5_list.append("-")

    metrics[f'{latest_year} TOP5品牌(占比)'] = top5_list

    # 格式化数字显示
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
    st.markdown("""
    **功能说明：**
    1. **数据源**：支持上传 CSV/Excel。
    2. **总销额计算**：严格取自源文件中的 **“市场整体”** 行，乘以系数后累加。
    3. **品牌合并**：自定义规则合并品牌数据（例如将“华为智选”并入“华为”）。
    """)

    with st.sidebar:
        st.header("1. 品牌合并规则")
        st.info("格式：目标品牌: 别名1, 别名2 (每行一个)")
        default_rules = """华为: 华为智选, 鸿蒙
paulmann: paulmann p
明基: benq, 麦朵尔"""
        rules_input = st.text_area("输入规则", value=default_rules, height=150)
        brand_mapping = parse_brand_rules(rules_input)

        st.header("2. 数据上传与配置")

        config = {'JD': {}, 'Tmall': {}}

        with st.expander("京东 (JD) 配置", expanded=True):
            st.markdown("**2024 年**")
            jd24_f = st.file_uploader("JD 2024 文件", type=['csv', 'xlsx'], key='jd24')
            jd24_c = st.number_input("JD 2024 系数", value=0.87, step=0.01, key='c_jd24')
            if jd24_f: config['JD']['2024'] = {'file': jd24_f, 'coeff': jd24_c}

            st.markdown("---")
            st.markdown("**2025 年**")
            jd25_f = st.file_uploader("JD 2025 文件", type=['csv', 'xlsx'], key='jd25')
            jd25_c = st.number_input("JD 2025 系数", value=0.87, step=0.01, key='c_jd25')
            if jd25_f: config['JD']['2025'] = {'file': jd25_f, 'coeff': jd25_c}

        with st.expander("天猫 (Tmall) 配置", expanded=True):
            st.markdown("**2024 年**")
            tm24_f = st.file_uploader("Tmall 2024 文件", type=['csv', 'xlsx'], key='tm24')
            tm24_c = st.number_input("Tmall 2024 系数", value=0.82, step=0.01, key='c_tm24')
            if tm24_f: config['Tmall']['2024'] = {'file': tm24_f, 'coeff': tm24_c}

            st.markdown("---")
            st.markdown("**2025 年**")
            tm25_f = st.file_uploader("Tmall 2025 文件", type=['csv', 'xlsx'], key='tm25')
            tm25_c = st.number_input("Tmall 2025 系数", value=0.72, step=0.01, key='c_tm25')
            if tm25_f: config['Tmall']['2025'] = {'file': tm25_f, 'coeff': tm25_c}

        run_btn = st.button("开始分析", type="primary", use_container_width=True)

    if run_btn:
        if not any(config['JD']) and not any(config['Tmall']):
            st.warning("请至少上传一个文件！")
            return

        st.subheader("分析结果")

        # 1. 总体合并表
        st.markdown("### 🏆 JD + Tmall 渠道汇总")
        df_combined, err = generate_analysis(config, brand_mapping)
        if df_combined is not None:
            st.dataframe(df_combined, use_container_width=True)
            csv = df_combined.to_csv(index=False).encode('utf-8-sig')
            st.download_button("下载汇总表 (CSV)", csv, "combined_analysis.csv", "text/csv")
        else:
            st.error(err)

        # 2. 分平台表
        if any(config['JD']):
            st.markdown("---")
            st.markdown("### 🐶 京东 (JD) 独立分析")
            jd_conf = {'JD': config['JD']}
            df_jd, _ = generate_analysis(jd_conf, brand_mapping)
            st.dataframe(df_jd, use_container_width=True)

        if any(config['Tmall']):
            st.markdown("---")
            st.markdown("### 🐱 天猫 (Tmall) 独立分析")
            tm_conf = {'Tmall': config['Tmall']}
            df_tm, _ = generate_analysis(tm_conf, brand_mapping)
            st.dataframe(df_tm, use_container_width=True)


if __name__ == "__main__":
    main()