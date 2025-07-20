## A Closer Look at Backdoor Attacks on CLIP ICML2025

### Requirements

Follow the code in [clip_text_span](https://github.com/yossigandelsman/clip_text_span) to create a Conda environment:
```bash
conda env create -f environment.yml
conda activate DBCLIP
```

### Reproduce

#### Obtain a backdoored CLIP

By following the code in [CleanCLIP](https://github.com/nishadsinghi/CleanCLIP), you can obtain a backdoored CLIP by fine-tuning the OpenAI open-sourced CLIP based on the generated poisoned data. Or directly downloading a backdoored CLIP checkpoint (banana_badclip_vitB32) at https://drive.google.com/file/d/1oAqydyqcWJvwc3ainzMDqpBJDJyPAUSW/view?usp=sharing.

Model dir: "/model_path"

#### Extract representations

clean representations: 
```bash
python extract_representations.py --pretrained {model_path} --datasets imagenet --backdoor_type badnet --patch_type random --patch_location random --target_label 954
```

Backdoor representations: 
```bash
python extract_representations.py --pretrained {model_path} --datasets imagenet --backdoor_type badnet --patch_type random --patch_location random --target_label 954 --add_backdoor
```

Classifier: 
```bash
python extract_classifier.py --pretrained {model_path} --datasets imagenet --backdoor_type badnet --target_label 954
```

Parameters of backdoors:

| Backdoor | Backdoor type | patch_type|patch_location|
|--------|--------|--------|--------|
|BadNet | badnet | random | random|
| Blended | blended | blended | blended|
| BadNetLC | badnet | random |random |
| ISSBA |issba  | issba |issba |
| BadCLIP | badclip | badclip | middle|

|Datasets|ImageNet-1K | Caltech-101|Oxford Pets |
|--------|--------|--------|--------|
|Class Name| banana | accordion  |  samoyed |
|Class Label|  954 |  3 | 29 |


This step saves the representation files to a path for use in the next step.

#### Run experiments

Ablation experiments in compute_bd_heads.py 

Forward ablation: calculate_mean_ablate_layer1
Backward ablation: calculate_mean_ablate_layer2
Separate ablation: calculate_mean_ablate_layer3

Decomp-Rep:

```bash
python compute_bd_heads.py ----datasets imagenet --backdoor_type badnet --target_label 954 --head_ablation mean_ablate --ablate_means edit
python compute_bd_heads.py ----datasets imagenet --backdoor_type blended --target_label 954 --mlp_ablation mean_ablate
```

Decomp-Det:
```bash
python compute_bd_heads.py ----datasets imagenet --backdoor_type badnet --target_label 954 --head_ablation mean_ablate --ablate_means detect
```

Ablation study:
reverse: 
```bash
python compute_bd_heads.py ----datasets imagenet --backdoor_type badnet --target_label 954 --head_ablation mean_ablate --ablate_means reverse
```

Abandon: 
```bash
python compute_bd_heads.py ----datasets imagenet --backdoor_type badnet --target_label 954 --head_ablation zero_value_ablate --ablate_means edit
```

Random prototypes: 
```bash
python compute_bd_heads.py ----datasets imagenet --backdoor_type badnet --target_label 954 --head_ablation random_value_ablate --ablate_means edit
```


