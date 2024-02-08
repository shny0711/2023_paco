import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# CSVファイルのリスト
csv_files = ['/home/shunya/dev/2023_11_paco/mixdata/csvdata/return_data_1.csv', '/home/shunya/dev/2023_11_paco/mixdata/csvdata/return_data_2.csv', '/home/shunya/dev/2023_11_paco/mixdata/csvdata/return_data_3.csv']

# データを格納するリスト
dataframes = []

# CSVファイルからデータを読み込む
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dataframes.append(df)

# データの平均と標準偏差を計算
mean_values = np.mean([df[['weight', 'reward', 'x']] for df in dataframes], axis=0)
std_values = np.std([df[['weight', 'reward', 'x']] for df in dataframes], axis=0)

mean_values = pd.DataFrame(mean_values, columns=["weight", "reward", "x"])
std_values = pd.DataFrame(std_values, columns=["weight", "reward", "x"])

# グラフを作成
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# 上のサブプロット：平均の折れ線グラフ
axes[0].plot(mean_values['weight'], mean_values['reward'], label='mixed policy', color='blue')
axes[0].fill_between(mean_values['weight'], mean_values['reward'] - std_values['reward'], mean_values['reward'] + std_values['reward'],
                     color='blue', alpha=0.2)
axes[0].set_xlabel('weight of 1.0')
axes[0].set_ylabel('return')

# 下のサブプロット：平均の折れ線グラフ
axes[1].plot(mean_values['weight'], mean_values['x'], label='mixed policy', color='green')
axes[1].fill_between(mean_values['weight'], mean_values['x'] - std_values['x'], mean_values['x'] + std_values['x'],
                     color='green', alpha=0.2)
axes[1].set_xlabel('weight of 1.0')
axes[1].set_ylabel('x')

axes[0].axhline(y=1503.993, color='red', linestyle='--', label="optimal policy")
axes[1].axhline(y=5.684, color='red', linestyle='--', label="optimal policy")


axes[0].legend()
axes[1].legend()

plt.tight_layout()
plt.savefig("mixdata/graphdata/newgraph.pdf")

# , label='Standard Deviation'
# fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# # 上のサブプロット：平均の折れ線グラフ
# axes[0].plot(df['weight'], df['reward'], label='return', color='blue')
# axes[0].plot(df[])
# axes[0].set_xlabel('weight of 1.0')
# axes[0].set_ylabel('return')
# axes[0].legend()

# # 下のサブプロット：平均の折れ線グラフ
# axes[1].plot(mean_values['weight'], mean_values['x'], label='Average x', color='green')
# axes[1].fill_between(mean_values['weight'], mean_values['x'] - std_values['x'], mean_values['x'] + std_values['x'],
#                      color='green', alpha=0.2, label='Standard Deviation')
# axes[1].set_xlabel('weight of 1.0')
# axes[1].set_ylabel('x')
# axes[1].legend()

# plt.tight_layout()
# plt.savefig("mixdata/graphdata/graph.png")