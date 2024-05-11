# Constrained Molecule Optimization

## Important notes
The original property JT-VAE was not compatible with the rest of the code. I rewrote it to follow the datahandling and the updated "fast" code.
It was not clear to me how the processing of SMILES to moltrees was refactored.
Instead of rewriting the existing datautils.py i created a separate file datautils_prop.py. In this way i avoided dealing with one script having to handle both the data for the regular JT-VAE without properties as training input and the data for JT_prop_vae where properties were given as input as well.


## Train conditional JT-VAE

The difference form the regular training, is now we need to supply a file with the same length as the training file, which contains the properties for each molecule.

To do training run the following scripts in a bash submission from the repo root directory:

```
python -u fast_jtnn/mol_tree.py -i $dataset -v $vocab
python -u fast_molopt/preprocess_prop.py --train $dataset --split 10 --jobs 8 --output $output --prop_path $prop_path
python -u fast_molopt/vae_train_prop.py --train $output --vocab $vocab --save_dir $save_dir
```

## Optimization in latent space


An example of running the optimization is:

```
python -u fast_molopt/optimize.py --training_path $dataset --vocab_path $vocab --cutoff 0.2 --lr 2 --model_path $model --prop_path $prop_path
```

## Specific Run-Through

```
python -u fast_jtnn/mol_tree.py -i data/labeled_set/train_full.txt -v vocabs/monodentate_conditional.txt
python -u fast_molopt/preprocess_prop.py --train data/labeled_set/train_full.txt --split 10 --jobs 8 --output data/labeled_set/preprocessed/ --prop_path data/labeled_set/train_prop_full.txt
python -u fast_molopt/vae_train_prop.py --train data/labeled_set/preprocessed/ --vocab vocabs/monodentate_conditional.txt --save_dir models/monodentate_conditional/
```

- (Make sure models are in right directory after training.)
- Sample 160 ligands with [this notebook](../sample_monodentates_from_regions_before_optimizing.ipynb).

```
python -u fast_molopt/optimize.py --training_path data/labeled_set/train_TL_sample.txt --vocab_path vocabs/monodentate_conditional.txt --cutoff 0.2 --lr 2 --model_path models/monodentate_conditional/model.epoch-149 --prop_path data/labeled_set/train_prop_TL_sample.txt
```
