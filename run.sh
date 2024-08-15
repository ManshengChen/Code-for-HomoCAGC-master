for seed in 0 1 2 3 4 5 6 7 8 9
do
  python main.py --seed $seed
done

echo "seed" >> training_log.txt
echo "" >> training_log.txt

for wd1 in 0.1 0.01 0.001 0.0001 0.00001
do
 python main.py --wd1 $wd1
done

echo "wd1" >> training_log.txt
echo "" >> training_log.txt

for der in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
 python main.py --der $der
done

echo "der" >> training_log.txt
echo "" >> training_log.txt

for dfr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
 python main.py --dfr $dfr
done

echo "dfr" >> training_log.txt
echo "" >> training_log.txt

for tau in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
 python main.py --tau $tau
done

echo "tau" >> training_log.txt
echo "" >> training_log.txt

for lr1 in 0.1 0.01 0.001 0.0001 0.00001
do
 python main.py --lr1 $lr1
done

echo "lr1" >> training_log.txt
echo "" >> training_log.txt

for hid_dim in 8 16 32 64 128 256 512
do
 python main.py --hid_dim $hid_dim
done

echo "hid_dim" >> training_log.txt
echo "" >> training_log.txt

for out_dim in 8 16 32 64 128 256 512
do
 python main.py --out_dim $out_dim
done

echo "out_dim" >> training_log.txt
echo "" >> training_log.txt

for n_layers in 2 3 4 5 6 7 8 9
do
 python main.py --n_layers $n_layers
done

echo "n_layers" >> training_log.txt
echo "" >> training_log.txt
