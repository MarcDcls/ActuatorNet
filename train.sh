wandb=1
epochs=300
nodes=(32 64 128 256)
windows=(2 5 10 20)
activations=("ReLU" "LeakyReLU" "Tanh" "Softsign")

for a in "${activations[@]}"; do
    for w in "${windows[@]}"; do
        for n in "${nodes[@]}"; do
            python learn.py -s $w -w $wandb -a $a -e $epochs -n $n
        done
    done
done
