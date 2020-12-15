data_folder=../q1_data
for i in 2 3 4 5
do
  for j in 1 2
  do
    python main.py --epochs 20 --tmax 20 --transform $i --task $j --root_dir $data_folder
  done
done

for j in 1 2
do
  python main.py --epochs 100 --tmax 100 --transform $i --task 1 --root_dir $data_folder
done
