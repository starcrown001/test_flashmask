import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import matplotlib.gridspec as gridspec
import pandas as pd
import glob

# 读取 TSV 格式的数据到 DataFrame
def read_tsv_to_dataframe(file_path):
    try:
        # 使用 pandas 的 read_csv 函数，并指定分隔符为制表符
        df = pd.read_csv(file_path, sep='\t')
        # 清理列名中的多余空格
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_bar(categories, save_path, baseline_key):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import font_manager as fm

    colors = ['#39CFC5', '#FF7D5E']
    font_prop = fm.FontProperties()
    plt.rcParams['axes.unicode_minus'] = False

    # 动态计算行数和列数
    num_categories = len(categories)
    num_cols = min(3, num_categories)
    num_rows = (num_categories + num_cols - 1) // num_cols

    fig = plt.figure(figsize=(6*num_cols, 6*num_rows))
    gs = gridspec.GridSpec(nrows=num_rows, ncols=num_cols)
    axs = []
    bar_height = 0.8

    for idx, (category, data) in enumerate(categories.items()):
        row = idx // num_cols
        col = idx % num_cols
        ax = fig.add_subplot(gs[row, col])
        axs.append(ax)

        labels = data['labels']
        baseline = data[baseline_key]
        flashmaskv3 = data['flashmaskv3']
        x = np.arange(len(labels))
        # 对于时间，改进是负百分比（时间减少）
        increments = [(fm - fa) / fa * 100 for fa, fm in zip(baseline, flashmaskv3)]

        # 绘制 baseline 柱状图
        if baseline_key == 'flashmaskv1':
            ax.barh(x, baseline, bar_height, label='FlashMask V1', color=colors[0])
        elif baseline_key == 'flexattention':
            ax.barh(x, baseline, bar_height, label='Flex Attention', color=colors[0])
        elif baseline_key == 'old_flashmaskv3':
            ax.barh(x, baseline, bar_height, label='Old FlashMask V3 (2 weeks ago)', color=colors[0])
        elif baseline_key == 'flashmaskv4':
            ax.barh(x, baseline, bar_height, label='Block Attention', color=colors[0])
        elif baseline_key == 'magiattention':
            ax.barh(x, baseline, bar_height, label='Magi Attention', color=colors[0])
        else:
            raise ValueError(f"baselinekey must be flashmaskv1, flexattention or old_flashmaskv3, got {baseline_key}")

        # 在 baseline 柱状图右端内部标注白色数字
        for j in range(len(labels)):
            ax.text(
                baseline[j] - max(baseline)*0.01, x[j],
                f'{baseline[j]:.1f}',
                va='center', ha='right', fontsize=12, color='white',
                fontproperties=font_prop
            )

        # 绘制 FlashMask V3 柱状图
        ax.barh(x, flashmaskv3, bar_height, label='FlashMask V3', color=colors[1])

        # 在 flashmaskv3 柱状图右端外部标注时间和改进百分比
        for j in range(len(labels)):
            increment = increments[j]
            sign = '' if increment < 0 else '+'
            ax.text(
                max(baseline[j], flashmaskv3[j]) + max(baseline)*0.005, x[j],
                f'{flashmaskv3[j]:.1f} ({sign}{increment:.1f}%)',
                va='center', ha='left', fontsize=12, color='black',
                fontproperties=font_prop
            )

        # Y轴
        ax.set_yticks(x)
        if idx == 0:
            ax.set_yticklabels(labels, fontsize=14, fontproperties=font_prop)
        else:
            ax.set_yticklabels(['' for _ in labels], fontsize=14, fontproperties=font_prop)
        ax.invert_yaxis()
        ax.set_xlabel(data['xlabel'], fontsize=14, fontproperties=font_prop)
        ax.set_title(category, fontsize=16, fontproperties=font_prop)
        ax.tick_params(axis='x', labelsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 图例
    handles, legend_labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc='upper center', ncol=2,
        prop=font_prop.copy().set_size(14), frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path+'.pdf', dpi=300, format='pdf')
    plt.show()


def main(baseline: str = "flashmaskv1"):
    plt.rcParams['font.family'] = "Liberation Mono"

    root_dir = '.'
    # for dtype in ['bf16', 'fp16']:
    for kernel in ["fwd", "bwd", "total"]:
        for dtype in ['bf16']:
            for headdim in [128]:
                categories = {}
                # for seqlen in [32768,131072]:
                for seqlen in [8192,32768,131072]:
                # for seqlen in [8192]:
                    method_to_df = {}
                    for method in [baseline, 'flashmaskv3']:
                        filenames = glob.glob(f'{root_dir}/{dtype}/{method}_*{seqlen}_*_{headdim}*.csv')
                        print(filenames)
                        dataframes = []
                        non_numeric_column = 'Operation'
                        if kernel == "fwd":
                            metric = 'FW Time (ms)'
                        elif kernel == "bwd":
                            metric = 'BW Time (ms)'
                        elif kernel == "total":
                            metric = 'TOTAL Time (ms)'
                        else:
                            raise ValueError(f"kernel must be fwd or bwd, but got {kernel}")

                        columns_to_average = [metric, 'Sparsity']

                        for file_path in filenames:
                            df = read_tsv_to_dataframe(file_path)
                            dataframes.append(df)

                        if len(dataframes) == 0:
                            print(f"No files found for {method} with seqlen {seqlen}")
                            method_to_df[method] = None
                            continue

                        aligned_dataframes = [df[columns_to_average] for df in dataframes]
                        combined_data = pd.concat(aligned_dataframes, axis=0, keys=range(len(dataframes)))
                        mean_df = combined_data.groupby(level=1).mean()
                        print(mean_df)
                        print(dataframes[0].keys())
                        mean_df[non_numeric_column] = dataframes[0][non_numeric_column]
                        mean_df = mean_df[[non_numeric_column] + columns_to_average]
                        method_to_df[method] = mean_df
                        print('='*20)
                        print(mean_df)

                    # 检查是否有有效的数据
                    if method_to_df.get(baseline) is None or method_to_df.get('flashmaskv3') is None:
                        print(f"Skipping seqlen {seqlen} due to missing data")
                        continue

                    one_item = {}
                    # 获取两个方法都有的共同操作
                    baseline_ops = set(method_to_df[baseline]['Operation'].tolist())
                    flashmaskv3_ops = set(method_to_df['flashmaskv3']['Operation'].tolist())
                    common_ops = sorted(list(baseline_ops & flashmaskv3_ops))

                    labels = [op.strip() for op in common_ops]
                    one_item['labels'] = labels

                    # 根据操作名称对齐数据
                    baseline_values = []
                    flashmaskv3_values = []
                    for op in common_ops:
                        baseline_idx = method_to_df[baseline][method_to_df[baseline]['Operation'] == op].index[0]
                        flashmaskv3_idx = method_to_df['flashmaskv3'][method_to_df['flashmaskv3']['Operation'] == op].index[0]
                        baseline_values.append(method_to_df[baseline].loc[baseline_idx, metric])
                        flashmaskv3_values.append(method_to_df['flashmaskv3'].loc[flashmaskv3_idx, metric])

                    one_item[baseline] = baseline_values
                    one_item['flashmaskv3 improvement'] = [fm - fa for fm, fa in zip(flashmaskv3_values, baseline_values)]
                    one_item['flashmaskv3'] = flashmaskv3_values
                    if kernel == "fwd":
                        one_item['xlabel'] = 'Fwd Time (ms)'
                    elif kernel == "bwd":
                        one_item['xlabel'] = 'Bwd Time (ms)'
                    elif kernel == "total":
                        one_item['xlabel'] = 'Total Time (ms)'
                    else:
                        raise ValueError(f"kernel must be fwd or bwd, but got {kernel}")

                    categories[f'Sequence length {seqlen//1024}K, head dim {headdim}'] = one_item
                plot_bar(categories, f'{root_dir}/flashmaskv3_vs_{baseline}_{dtype}_{headdim}_{kernel}_time', baseline)

if __name__ == "__main__":
    from jsonargparse import ArgumentParser
    parser = ArgumentParser(description="Run specific examples or all examples.")

    parser.add_argument(
        "--baseline",
        type=str,
        default="flashmaskv1"
    )

    args = parser.parse_args()
    main(**vars(args))
