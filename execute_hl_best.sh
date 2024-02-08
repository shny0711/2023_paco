 #  Fric1.0
path_H_3="/home/shunya/dev/2023_11_paco/data/newMarathonFric1.0-v0/MLPSAC/20240119_023028/best"

#Fric0.01
path_L_3="/home/shunya/dev/2023_11_paco/data/newMarathonFric0.01-v0/MLPSAC/20240119_212046/best"

step=100000

# # 1
python train_hlsac.py --path1 $path_H_3 --path2 $path_L_3 --all-step $((3*step))
python train_hlsac.py --path1 $path_H_3 --path2 $path_L_3 --all-step $((3*step))
# 12
python train_mhlsac.py --path1 $path_H_3 --path2 $path_L_3 --all-step $((3*step))
python train_mhlsac.py --path1 $path_H_3 --path2 $path_L_3 --all-step $((3*step))

# 3
python train_mhlsac2.py --path1 $path_H_3 --path2 $path_L_3 --all-step $((3*step))
python train_mhlsac2.py --path1 $path_H_3 --path2 $path_L_3 --all-step $((3*step))

# 4
python train_mhlsac3.py --path1 $path_H_3 --path2 $path_L_3 --all-step $((3*step))
python train_mhlsac3.py --path1 $path_H_3 --path2 $path_L_3 --all-step $((3*step))