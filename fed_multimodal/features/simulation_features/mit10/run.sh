for alpha in 0.1; do
    for mm_rate in 0.25 0.5; do
        taskset 50 python3 simulation_feature.py --alpha $alpha --en_missing_modality --missing_modailty_rate $mm_rate
    done
done

for alpha in 0.1; do
    for label_noisy in 0.2 0.4; do
        taskset 50 python3 simulation_feature.py --alpha $alpha --en_label_nosiy --label_nosiy_level $label_noisy
    done
done