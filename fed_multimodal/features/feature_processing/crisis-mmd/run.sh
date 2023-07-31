# extract mobilenet_v2 feature
taskset 100 python3 extract_frame_feature.py --feature_type mobilenet_v2 --alpha 1.0

# extract mfcc feature
taskset 100 python3 extract_audio_feature.py --feature_type mfcc --alpha 1.0



