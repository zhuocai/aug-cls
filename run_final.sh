data_folder=../q1_data
for j in 1 2
do
  python main.py --epochs 100 --tmax 100 --transform 1 --task $j --root_dir $data_folder
done
