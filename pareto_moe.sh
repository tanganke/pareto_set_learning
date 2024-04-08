function train(){
    python scripts/clip_pareto_${method}.py train=true evaluate=false \
        version=${version} test_datasets=${test_datasets} model=${model} partial=$partial \
        alpha=1 lr=1e-2 num_devices=$num_devices batch_size=${batch_size} num_steps=4000
}

function evaluate() {
    python scripts/clip_pareto_${method}.py train=false evaluate=true batch_size=64 \
        version=${version} test_datasets=${test_datasets} model=${model} partial=$partial
}

function run() {
    train
    evaluate
}

function run_all() {
    num_devices=1
    # ViT-B-32 Ours (MLP only)
    model=ViT-B-32
    partial=true

    batch_size=16

    # two tasks
    version=0
    test_datasets="[SUN397,Cars]"
    run

    version=1
    test_datasets="[SUN397,DTD]"
    run

    version=2
    test_datasets="[Cars,DTD]"
    run

    # three tasks
    version=3
    test_datasets="[SUN397,Cars,DTD]"
    run
    
    # eight tasks
    version=4
    test_datasets="[SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD]"
    train
    python scripts/clip_pareto_${method}.py train=false evaluate=true batch_size=64 \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial num_evaluation_samples=equal_weight

    # --------------------
    # ViT-B-32 Ours (All layers)
    partial=false

    # two tasks
    version=5
    test_datasets="[SUN397,Cars]"
    run

    version=6
    test_datasets="[SUN397,DTD]"
    run

    version=7
    test_datasets="[Cars,DTD]"
    run

    # three tasks
    version=8
    test_datasets="[SUN397,Cars,DTD]"
    run

    # eight tasks
    version=9
    test_datasets="[SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD]"
    train
    python scripts/clip_pareto_${method}.py train=false evaluate=true batch_size=64 \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial num_evaluation_samples=equal_weight

    # ViT-L-14
    model=ViT-L-14
    partial=true
    version=10
    test_datasets="[SUN397,Cars]"
    run
    
    version=11
    test_datasets="[SUN397,DTD]"
    run

    version=12
    test_datasets="[Cars,DTD]"
    run

    version=14
    test_datasets="[SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD]"
    num_devices=4
    batch_size=12
    CUDA_VISIBLE_DEVICES=4,5,6,7 train
    python scripts/clip_pareto_${method}.py train=false evaluate=true batch_size=64 \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial num_evaluation_samples=equal_weight


    partial=false
    version=15
    batch_size=16
    test_datasets="[SUN397,Cars]"
    run

    version=16
    test_datasets="[SUN397,DTD]"
    run

    version=17
    test_datasets="[Cars,DTD]"
    run

    version=19
    num_devices=4
    batch_size=12
    test_datasets="[SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD]"
    CUDA_VISIBLE_DEVICES=4,5,6,7 train
    python scripts/clip_pareto_${method}.py train=false evaluate=true batch_size=64 \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial num_evaluation_samples=equal_weight
}


method=moe_ls
run_all

method=moe_epo
run_all
