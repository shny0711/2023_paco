import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def read_csv(filepath):
    """
    CSVファイルを読み込み、平均値の列のみを抽出する関数。
    """
    # CSVファイルを読み込む
    df_raw = pd.read_csv(filepath)
    # 'MAX''MIN'を含む列を除外
    df_step = df_raw['Step']
    df = df_raw[df_raw.columns[~df_raw.columns.str.contains('MAX|MIN|Step')]]
    return df_step, df
def calculate_stats(data, alpha=0.0):
    """
    標準誤差を計算したあとフィルタをかける．
    平均値、標準偏差、標準誤差を計算する関数。
    移動指数平均の係数alphaを指定可能。
    """
    mean = data.mean(axis='columns')
    if alpha > 0.0:
        # 移動指数平均を計算
        mean = mean.ewm(alpha=alpha).mean()
    return mean
def plot_axes(ax, filepaths, labels, alpha=0.0, error_type='std_dev', ylabel="Value", title=None):
    """
    複数のCSVファイルから時系列グラフをプロットする関数。
    error_typeは'std_dev'または'std_err'を指定可能。
    """
    for filepath, label in zip(filepaths,labels):
        # csvを読み出し
        step, data = read_csv(filepath)
        # 平均値を計算(alphaを調整すればフィルタがかかる)
        mean= calculate_stats(data, alpha)
        # グラフをプロット
        ax.plot(step, mean, label=label)
    # x軸のラベルを設定
    ax.set_xlabel('TimeSteps',fontsize=20)
    # y軸のラベルを設定
    ax.set_ylabel(ylabel,fontsize=20)
    # タイトルを設定
    if title != None:
        ax.set_title(title,fontsize=24)
    # 凡例を設定
    ax.legend(fontsize=16)
    # x,y軸の目盛りの数字のサイズを調整
    ax.tick_params(labelsize=16)
def plot_data(configs, sbp_size, save_name="img.png"):
    '''
    configs<list> :
        Key一覧
            filepaths<list> :  plotするcsvファイルの一覧
            alpha<float> : smoothing係数
            error_type<str> : エラーバーが'std_err'か'std_dev'かを選択．
            ylabel<str> : グラフのy軸のラベル
            title<str> : グラフのタイトルで，Noneでもよい
    sbp_size<taple> : size of subplots
    '''
    # 複数のグラフを一つの図に収めるための設定
    fig, axes=plt.subplots(*sbp_size,figsize=(sbp_size[1]*8, sbp_size[0]*6))
    if len(configs) > 1:
        axes = axes.flatten()
    else:
        axes = np.array([axes]) ## make axes subscriptable
    for ax, config in zip(axes,configs):
        # 各グラフをプロット
        plot_axes(ax,**config)
    fig.tight_layout()
    fig.savefig(f"mixdata/graphdata/{save_name}")
if __name__ == "__main__":
    alpha = 0.0
    # Hardware&Turn90 Other valve Error
    configs2 = []
    ### Graph1
    filepaths = ["/home/shunya/dev/2023_11_paco/csvdata/graphs/wandb_export_2024-01-23T22_27_10.599+09_00.csv","/home/shunya/dev/2023_11_paco/csvdata/graphs/wandb_export_2024-01-23T22_28_15.619+09_00.csv"]
    # labelsのlenはfilepathのlenと同じにする。
    labels = ["SAC", "SAC2"]
    configs2.append(
        {"filepaths":filepaths,"labels":labels, "alpha":alpha, "error_type":'std_err', "ylabel":"ValveError", "title":"ValveType:Key"}
    )
    ##  Graph2
    filepaths = ["/home/shunya/dev/2023_11_paco/csvdata/graphs/wandb_export_2024-01-23T22_27_10.599+09_00.csv","/home/shunya/dev/2023_11_paco/csvdata/graphs/wandb_export_2024-01-23T22_28_15.619+09_00.csv"]
    # labelsのlenはfilepathのlenと同じにする。
    labels = ["SAC", "SAC2"]
    configs2.append(
        {"filepaths":filepaths,"labels":labels, "alpha":alpha, "error_type":'std_err', "ylabel":"ValveError", "title":"ValveType:Key"}
    )
    # (2,1) : グラフの配置
    plot_data(configs2, (1,2),"HardwareTurn90_ValveError.pdf")