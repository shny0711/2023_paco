 #  Fric1.0
path_H_1="/home/shunya/dev/2023_11_paco/data/MarathonFric1.0-v0/MLPSAC/20240118_134921/best"
path_H_2="/home/shunya/dev/2023_11_paco/data/newMarathonFric1.0-v0/MLPSAC/20240118_200733/best"
path_H_3="/home/shunya/dev/2023_11_paco/data/newMarathonFric1.0-v0/MLPSAC/20240119_023028/best"

#Fric0.01
path_L_1="/home/shunya/dev/2023_11_paco/data/newMarathonFric0.01-v0/MLPSAC/20240119_085151/best"
path_L_2="/home/shunya/dev/2023_11_paco/data/newMarathonFric0.01-v0/MLPSAC/20240119_150719/best"
path_L_3="/home/shunya/dev/2023_11_paco/data/newMarathonFric0.01-v0/MLPSAC/20240119_212046/best"

# # 1
# python train_hlsac.py --path1 $path_H_1 --path2 $path_L_1
# python train_hlsac.py --path1 $path_H_2 --path2 $path_L_2
# python train_hlsac.py --path1 $path_H_3 --path2 $path_L_3

# 12
# python train_mhlsac.py --path1 $path_H_1 --path2 $path_L_1
# python train_mhlsac.py --path1 $path_H_2 --path2 $path_L_2
# python train_mhlsac.py --path1 $path_H_3 --path2 $path_L_3

# 3
python train_mhlsac2.py --path1 $path_H_1 --path2 $path_L_1
python train_mhlsac2.py --path1 $path_H_2 --path2 $path_L_2
python train_mhlsac2.py --path1 $path_H_3 --path2 $path_L_3

# 4
python train_mhlsac3.py --path1 $path_H_1 --path2 $path_L_1
python train_mhlsac3.py --path1 $path_H_2 --path2 $path_L_2
python train_mhlsac3.py --path1 $path_H_3 --path2 $path_L_3