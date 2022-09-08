# Robust fine-tuning

"Adversarial robustness against multiple $l_p$-threat models at the price of one and how to quickly fine-tune robust models to another threat model"\
*Francesco Croce, Matthias Hein*\
[https://arxiv.org/abs/2105.12508](https://arxiv.org/abs/2105.12508)

We propose to *i)* use adversarial training wrt Linf and L1 (alternating the two threat models) to achieve robustness also to L2 and *ii)* fine-tune models robust in one
Lp-norm to get multiple norm robustness or robustness wrt another Lq-norm.

## Code
### Training code

The file `train.py` allows to train or fine-tune models. For adversarial training use `--attack=apgd`, otherwise standard training is performed. The main arguments
for adversarial training are (other options in `train.py`)
+ `--l_norms='Linf L1'`, the list (as string with blank space separated items) of Lp-norms, even just one, to use for training (note that the training cost is the same
regardless of the number of threat models used),
+ `--l_eps`, list of thresholds epsilon for each threat model for training (if not given, the default values are used), sorted as the corresponding norms.
+ `--l_iters`, list of iterations in adversarial training for each threat model (possibly different), or `--at_iter`, number of steps for all threat models.

For training new models a PreAct ResNet-18 is used, by default with softplus activation function. 


### Fine-tuning existing models

To fine-tune a model add the `--finetune_model` flag, `--lr-schedule=piecewise-ft` to set the standard learning rate schedule,
`--model_dir=/path/to/pretrained/models` where to download or find the models.

+ We provide [here](https://drive.google.com/drive/folders/1hYWHp5UbTAm9RhSb8JkJZtcB0LDZDvkT?usp=sharing) pre-trained ResNet-18 robust wrt Linf, L2 and L1,
which can be loaded specifying `--model_name=pretr_L*.pth` (insert the desired norm).
+ It is also possible to use models from the [Model Zoo](https://github.com/RobustBench/robustbench#model-zoo) of [RobustBench](https://robustbench.github.io/)
with `--model_name=RB_{}` inserting the identifier of the classifier from the Model Zoo (these are automatically downloaded). Note that models trained with extra data should be fine-tuned with the same
(currently not supported in the code).

### Evaluation code
With `--final_eval` our standard evaluation (with APGD-CE and APGD-T, for a total of 10 restarts of 100 steps) is run for all threat models at the end of training.
Specifying `--eval_freq=k` a fast evaluation is run on test and training points every `k` epochs.

To evaluate a trained model one can run `eval.py` with `--model_name` as above for the pretrained model or `--model_name=/path/to/checkpoint/` for new or fine-tuned
classifiers. If the run has the automatically generated name, the corresponding architecture is loaded. More details about the options for evaluation in `eval.py`.

## Credits
Parts of the code in this repo is based on
+ [https://github.com/tml-epfl/understanding-fast-adv-training](https://github.com/tml-epfl/understanding-fast-adv-training)
+ [https://github.com/locuslab/robust_overfitting](https://github.com/locuslab/robust_overfitting)
