task=('A2C' 'A2P' 'A2R' 'C2A' 'C2P' 'C2R' 'P2A' 'P2C' 'P2R' 'R2A' 'R2C' 'R2P')
source=('Art' 'Art' 'Art' 'Clipart' 'Clipart' 'Clipart' 'Product' 'Product' 'Product' 'RealWorld' 'RealWorld' 'RealWorld')
target=('Clipart' 'Product' 'RealWorld' 'Art' 'Product' 'RealWorld' 'Art' 'Clipart' 'RealWorld' 'Art' 'Clipart' 'Product')

net='ResNet50'
dset='office-home'
factor='10'
up_rate='0.2'
agg_up='0.9'
pm_meth='500'
seed=('42' '43' '44')
index='0'
num_iter='5004'

echo $index

for((run_id=0; run_id < 10; run_id++))
  do	  
    echo ">> Seed 42: traning task ${index} : ${task[index]}"
    s_dset_path='data/OfficeHome/'${source[index]}'.txt'
    t_dset_path='data/OfficeHome/'${target[index]}'.txt'
    output_dir='officeHome/'${seed[0]}'/'${task[index]}
    python train.py \
       --net ${net} \
       --dset ${dset} \
       --factor ${factor} \
       --s_dset_path ${s_dset_path} \
       --t_dset_path ${t_dset_path} \
       --output_dir ${output_dir} \
       --update ${up_rate} \
       --pm_meth ${pm_meth} \
       --run_id ${run_id} \
       --agg_up ${agg_up} \
       --num_iter ${num_iter} \
       --seed ${seed[0]} &
    sleep 10
    echo ">> Seed 43: traning task ${index} : ${task[index]}"
    output_dir='officeHome/'${seed[1]}'/'${task[index]}
    python train.py \
       --net ${net} \
       --dset ${dset} \
       --factor ${factor} \
       --s_dset_path ${s_dset_path} \
       --t_dset_path ${t_dset_path} \
       --output_dir ${output_dir} \
       --update ${up_rate} \
       --pm_meth ${pm_meth} \
       --run_id ${run_id} \
       --agg_up ${agg_up} \
       --num_iter ${num_iter} \
       --seed ${seed[1]} &
    sleep 10
    echo ">> Seed 44: traning task ${index} : ${task[index]}"
    output_dir='officeHome/'${seed[2]}'/'${task[index]}
    python train.py \
       --net ${net} \
       --dset ${dset} \
       --factor ${factor} \
       --s_dset_path ${s_dset_path} \
       --t_dset_path ${t_dset_path} \
       --output_dir ${output_dir} \
       --update ${up_rate} \
       --pm_meth ${pm_meth} \
       --run_id ${run_id} \
       --agg_up ${agg_up} \
       --num_iter ${num_iter} \
       --seed ${seed[2]} 
    wait
done
