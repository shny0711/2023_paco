import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--readpath")
parser.add_argument("--savepath1")
parser.add_argument("--savepath")
args = parser.parse_args()


df = pd.read_csv(args.readpath)


# no fix
x1 = df["weight"]
y1 = df["reward"]
z1 = df["x"]

fig1, ax1 =plt.subplots(1,2)

ax1[0].set_xlabel("weight of env1.0")
ax1[0].set_ylabel("return")

ax1[1].set_xlabel("weight of env1.0")
ax1[1].set_ylabel("x_dist")

ax1[0].plot(x1,y1, lw=0.3)
ax1[1].plot(x1,z1, lw=0.3)

plt.savefig(args.savepath1)



for i in range(1000):
    if df.at[i, "reward"]<=0:
        df.at[i, "reward"]=0


# fix
x = df["weight"]
y = df["reward"]
z = df["x"]

fig, ax =plt.subplots(1,2)

ax[0].set_xlabel("weight of env1.0")
ax[0].set_ylabel("return")

ax[1].set_xlabel("weight of env1.0")
ax[1].set_ylabel("x_dist")

ax[0].plot(x,y, lw=0.3)
ax[1].plot(x,z, lw=0.3)

plt.savefig(args.savepath)


