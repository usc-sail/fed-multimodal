for ml_rate in 0.1 0.2 0.3 0.4 0.5; do
   python3 simulation_feature.py --alpha 0.1 --en_missing_label --missing_label_rate $ml_rate
done
