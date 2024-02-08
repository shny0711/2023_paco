 #  Fric1.0
path_H_1="/home/shunya/dev/2023_11_paco/data/MarathonFric1.0-v0/MLPSAC/20240118_134921/best"
path_H_3="/home/shunya/dev/2023_11_paco/data/newMarathonFric1.0-v0/MLPSAC/20240119_023028/best"

#Fric0.01
path_L_1="/home/shunya/dev/2023_11_paco/data/newMarathonFric0.01-v0/MLPSAC/20240119_085151/best"
path_L_3="/home/shunya/dev/2023_11_paco/data/newMarathonFric0.01-v0/MLPSAC/20240119_212046/best"

path_H_2="/home/shunya/dev/2023_11_paco/data/newMarathonFric1.0-v0/MLPSAC/20240118_200733/best"
path_L_2="/home/shunya/dev/2023_11_paco/data/newMarathonFric0.01-v0/MLPSAC/20240119_150719/best"

# python moe_movie.py --path1 $path_H_1 --path2 $path_L_1 --weight 70
# python moe_movie.py --path1 $path_H_3 --path2 $path_L_3 --weight 70
# python moe_movie.py --path1 $path_H_3 --path2 $path_L_3 --weight 400
# python moe_movie.py --path1 $path_H_3 --path2 $path_L_3 --weight 800

python moe_movie.py --path1 $path_H_2 --path2 $path_L_2 --weight 34
python moe_movie.py --path1 $path_H_2 --path2 $path_L_2 --weight 32
python moe_movie.py --path1 $path_H_2 --path2 $path_L_2 --weight 30
python moe_movie.py --path1 $path_H_2 --path2 $path_L_2 --weight 36