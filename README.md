Code of "A Closer Look at Backdoor Attacks on CLIP" ICML2025.

1. Obtain a backdoored CLIP

One can do this step by fine-tuning the OpenAI open-sourced CLIP by the generated poisoned data. (following the code in CleanCLIP)

Or directly downloading a backdoored CLIP checkpoint (banana_badclip_vitB32) at https://drive.google.com/file/d/1oAqydyqcWJvwc3ainzMDqpBJDJyPAUSW/view?usp=sharing

Model dir: "/model_path"

2. Extract representations


clean: python extract_representations.py --pretrained {model_path} --datasets imagenet --backdoor_type badnet --patch_type random --patch_location random --target_label 954

Backdoor: python extract_representations.py --pretrained {model_path} --datasets imagenet --backdoor_type badnet --patch_type random --patch_location random --target_label 954 --add_backdoor

Classifier: python extract_classifier.py --pretrained {model_path} --datasets imagenet --backdoor_type badnet --target_label 954

Parameter:

Backdoor:          BadNet Blended BadNetLC ISSBA  BadCLIP

Backdoor type:     badnet blended badnet   issba  badclip

patch_type:        random blended random   issba  bad clip

patch_location:    random blended random   issue  middle


Datasets:

ImageNet-1K banana 954
Caltech-101 accordion 3
Oxford Pets: samoyed 29


Please carefully check the args.

3. Run exp

Ablation experiments in compute_bd_heads.py 

Forward ablation: calculate_mean_ablate_layer1
Backward ablation: calculate_mean_ablate_layer2
Separate ablation: calculate_mean_ablate_layer3

Decomp-Rep:

python compute_bd_heads.py ----datasets imagenet --backdoor_type badnet --target_label 954 --head_ablation mean_ablate --ablate_means edit

python compute_bd_heads.py ----datasets imagenet --backdoor_type blended --target_label 954 --mlp_ablation mean_ablate

Decomp-Det:

python compute_bd_heads.py ----datasets imagenet --backdoor_type badnet --target_label 954 --head_ablation mean_ablate --ablate_means detect

Ablation study:

reverse: python compute_bd_heads.py ----datasets imagenet --backdoor_type badnet --target_label 954 --head_ablation mean_ablate --ablate_means reverse

Abandon: python compute_bd_heads.py ----datasets imagenet --backdoor_type badnet --target_label 954 --head_ablation zero_value_ablate --ablate_means edit

Random prototypes: python compute_bd_heads.py ----datasets imagenet --backdoor_type badnet --target_label 954 --head_ablation random_value_ablate --ablate_means edit


Please carefully check the args.


