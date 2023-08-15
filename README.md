# OneWORD: Adversarial Text Detection and Prediction Restoration Using One-Word Perturbation

## Dependencies

* TextAttack framework (https://github.com/QData/TextAttack) (install by : `pip install textattack` for basic packages (such as PyTorch and Transformers) or `pip install textattack [tensorflow,optional]` for tensorflow and optional dependences. Please refer TextAttack installation (https://textattack.readthedocs.io/en/latest/0_get_started/installation.html) for more detail.

## Usage

* Run the following commmands for corresponding objectives:
 `python3 detection.py`
 `python3 restoration.py`
 
### Parameters

#### Parameters for all objectives:

* `dataset` : dataset name (default = 'sst2'), complied with the names from HuggingFaceDataset (https://github.com/huggingface/datasets/tree/master/datasets) such as 'imdb', 'mrpc', 'yelp.'
* `attack`: attack is used for generating adversarial text (default = 'pwws'), compied with the names from TextAttack. Other attack names can be found in the TextAttack framework (https://textattack.readthedocs.io/en/latest/3recipes/attack_recipes_cmd.html#attacks-and-papers-implemented-attack-recipes-textattack-attack-recipe-recipe-name)
* `target`: name of target model (default = 'cnn-sst2'), complied with the names from TextAttack. Other model name can be found in the TextAttack framework (https://textattack.readthedocs.io/en/latest/3recipes/models.html#textattack-models)
* `attacks_for_defense`: attacks is used for OneWORD (default = ['pwws', 'deepwordbug'])
* `supporters`: list of support-model names (default = ['roberta-base-sst2'])
* `num_train`: number of testing samples (default = 1000)
* `num_dev`:  number of validation samples (default = 100)
* `num_test`: number of testing samples (default = 100)

### Examples

* Running with default parameters : `python3 detection.py`
* Running with customized parameters : `python3 restoration.py --dataset imdb --attack deepwordbug --target cnn-imdb --auxiliary_attacks deepwordbug pwws --num_train 500 --num_dev 50 --num_test 50 --supporters lstm-imdb bert-base-uncased-imdb`

### Acknowledgement
* TextAttack framework (https://github.com/QData/TextAttack)

