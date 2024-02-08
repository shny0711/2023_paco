 #  Fric1.0
path_H_1="/home/shunya/dev/2023_11_paco/data/MarathonFric1.0-v0/MLPSAC/20240118_134921/best"
path_H_2="/home/shunya/dev/2023_11_paco/data/newMarathonFric1.0-v0/MLPSAC/20240118_200733/best"
path_H_3="/home/shunya/dev/2023_11_paco/data/newMarathonFric1.0-v0/MLPSAC/20240119_023028/best"

#Fric0.01
path_L_1="/home/shunya/dev/2023_11_paco/data/newMarathonFric0.01-v0/MLPSAC/20240119_085151/best"
path_L_2="/home/shunya/dev/2023_11_paco/data/newMarathonFric0.01-v0/MLPSAC/20240119_150719/best"
path_L_3="/home/shunya/dev/2023_11_paco/data/newMarathonFric0.01-v0/MLPSAC/20240119_212046/best"




#Fric0.1
# path_H_1
# path_H_2
# path_H_3

# predeict_error
# python enjoy_moe.py --path1 $path_H_1 --path2 $path_L_3
# python enjoy_moe.py --path1 $path_H_2 --path2 $path_L_3
# python enjoy_moe.py --path1 $path_H_3 --path2 $path_L_2


#savepath
save_1="mixdata/csvdata/return_data_1.csv"
save_2="mixdata/csvdata/return_data_2.csv"
save_3="mixdata/csvdata/return_data_3.csv"



# exxploring of weight
# # output data of return
# python enjoy_moe-test.py --path1 $path_H_1 --path2 $path_L_1 --savepath $save_1
# python enjoy_moe-test.py --path1 $path_H_2 --path2 $path_L_2 --savepath $save_2
# python enjoy_moe-test.py --path1 $path_H_3 --path2 $path_L_3 --savepath $save_3


# no fix
graph1_1="mixdata/graphdata/return_data_1.eps"
graph1_2="mixdata/graphdata/return_data_2.eps"
graph1_3="mixdata/graphdata/return_data_3.eps"

# fix
graph_1="mixdata/graphdata/fix_return_data_1.eps"
graph_2="mixdata/graphdata/fix_return_data_2.eps"
graph_3="mixdata/graphdata/fix_return_data_3.eps"




# output graph
python enjoy_testgraph.py --readpath $save_1 --savepath1 $graph1_1 --savepath $graph_1
python enjoy_testgraph.py --readpath $save_2 --savepath1 $graph1_2 --savepath $graph_2
python enjoy_testgraph.py --readpath $save_3 --savepath1 $graph1_3 --savepath $graph_3
