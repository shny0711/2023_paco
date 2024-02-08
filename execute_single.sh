 #  Fric1.0
path_H_1="/home/shunya/dev/2023_11_paco/data/MarathonFric1.0-v0/MLPSAC/20240118_134921/best"
path_H_2="/home/shunya/dev/2023_11_paco/data/newMarathonFric1.0-v0/MLPSAC/20240118_200733/best"
path_H_3="/home/shunya/dev/2023_11_paco/data/newMarathonFric1.0-v0/MLPSAC/20240119_023028/best"

#Fric0.01
path_L_1="/home/shunya/dev/2023_11_paco/data/newMarathonFric0.01-v0/MLPSAC/20240119_085151/best"
path_L_2="/home/shunya/dev/2023_11_paco/data/newMarathonFric0.01-v0/MLPSAC/20240119_150719/best"
path_L_3="/home/shunya/dev/2023_11_paco/data/newMarathonFric0.01-v0/MLPSAC/20240119_212046/best"

#Fric0.1
path_M_1="/home/shunya/dev/2023_11_paco/data/newMarathonFric0.1-v0/MLPSAC/20240120_034216/best"
path_M_2="/home/shunya/dev/2023_11_paco/data/newMarathonFric0.1-v0/MLPSAC/20240120_095830/best"
path_M_3="/home/shunya/dev/2023_11_paco/data/newMarathonFric0.1-v0/MLPSAC/20240121_012048/best"

env_H="MarathonFric1.0-v0"
env_L="MarathonFric0.01-v0"
env_M="MarathonFric0.1-v0"


python enjoy_moe.py --path1 $path_H_1 --path2 "a" --env $env_H
python enjoy_moe.py --path1 $path_H_2 --path2 "a" --env $env_H
python enjoy_moe.py --path1 $path_H_3 --path2 "a" --env $env_H

python enjoy_moe.py --path1 $path_L_1 --path2 "b" --env $env_L
python enjoy_moe.py --path1 $path_L_2 --path2 "b" --env $env_L
python enjoy_moe.py --path1 $path_L_3 --path2 "b" --env $env_L

# python enjoy_moe.py --path1 $path_M_1 --path2 "c" #--env $env_M
# python enjoy_moe.py --path1 $path_M_2 --path2 "c" #--env $env_M
# python enjoy_moe.py --path1 $path_M_3 --path2 "c" #--env $env_M