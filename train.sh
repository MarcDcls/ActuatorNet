wandb=1
epochs=500
windows=(2 5 10)
activations=("ReLU" "LeakyReLU" "Tanh" "Softsign")

python learn.py -s 10 -w $wandb -a Tanh -e $epochs

for a in "${activations[@]}"; do
    for w in "${windows[@]}"; do
        python learn.py -s $w -w $wandb -a $a -e $epochs
    done
done
