Reimplementation of CNN-based contact prediction models

DeepCov:
[Fully convolutional neural networks for protein residue-residue contact prediction
David T. Jones and Shaun M. Kandathil - University College London](https://github.com/psipred/DeepCov)

DeepCon:
[Dilated convolution network with dropout (best reported performing model, Fig.3d)](https://github.com/ba-lab/DEEPCON/)

## Datasets
Get train & test data from [Bioinformatics Group, UCL DEPARTMENT OF COMPUTER SCIENCE](http://bioinfadmin.cs.ucl.ac.uk/downloads/contact_pred_datasets/)

## Data structure
```console
DeepCov/
    setup.sh # script to compile cov21stats 
    feature_extraction.ipynb # script that calls cov21stats to extract features for a dataset
    train.py # training script
    models.py # models
    data.py   # data loader
    predict.ipynb # make prediction on test data with a trained model
    evaluate.ipynb # evaluation 
    read_result.ipynb # read evaluation results
    
    src/ 
       /cov21stats.c  # C source code for covariance stats computation
    
    bin/ 
       /cov21stats  # compiled covariance stats
    
    data/
        train/
            aln/  # contains 3456 aligments files
            21c/  # contains 3456 feature files, each in shape (441, m, m)
            map/  # ground truth
        test/
            psicov150/
                     aln/
                     21c/
                     pdb/ # ground truth
                     rr/  # predicted contact maps
```

## Data preparation
### Compile feature extractor
Get the scripts setup.sh and cov21stats at https://github.com/psipred/DeepCov
Run setup.sh to compile the extractor:
```console
./setup.sh
```

### Run feature extraction
Run once:
```python
feature_extraction.ipynb
```

## Run training 
- For the first time training, run:
```python
python train.py [--model=DeepCon] [--gpu=1]
```

- In case of resume training, specified the saved checkpoint file:

```python
python train.py [--model=DeepCon] [--gpu=1] [--resume=DeepCov_checkpoint.pth.tar]
```

## Prediction on test data
- Modify the path to prediction folder if needed (default: 'data/test/psicov150/rr')
- Run
```Python
predict.ipynb
```

## Evaluation
- Modify the path to prediction folder if needed (default: 'data/test/psicov150/rr')
- Run
```Python
evaluate.ipynb
```

## Read results (precision in long-range distance: P@5, P@L/10, P@L/5, P@L/2, P@L)
- Modify the result file name
- Run
```Python
read_result.ipynb
```
