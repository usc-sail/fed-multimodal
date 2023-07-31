
for alpha in 0.1; do
    for label_noisy in 0.2 0.4; do
        taskset 50 python3 simulation_feature.py --en_label_nosiy --label_nosiy_level $label_noisy
    done
done