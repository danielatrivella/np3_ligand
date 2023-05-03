# NP³ DL Segmentation: A Deep Learning Pipeline for the Semantic Segmentation of Lig-PCDB

  This repository contains the code for the DL models training pipeline and validation (step 6 from part B and part C of the workflow presented in np3_Lig-PCDB).

  Four labeling approaches were validated by training good performance DL models for the semantic 
segmentation of a stratified training dataset of Lig-PCDB. 
The average performance in the cross-validation of their best DL model is presented below using the mIoU, mF1, Precision and Recall metrics. 

| DL Model         | dmax | Loss  weights             | Epochs | Test mIoU | F1 score | Precision | Recall |
|------------------|------|---------------------------|--------|-----------|----------|-----------|--------|
| LigandRegion     | 1    | 1,2.5                     | 120    | 77.4      | 87.0     | 86.5      | 87.4   |
| AtomCycle        | 1.4  | 1,2.5,2.5                 | 120    | 71.0      | 82.5     | 80.5      | 84.9   |
| AtomC347CA56     | 865  | 1,10,5,5,50,5,500,500,500 | 200    | 49.7      | 62.4     | 58.2      | 74.1   |
| AtomSymbolGroups | 81.5 | 16,16,44,108,853          | 160    | 59.0      | 73.1     | 68.6      | 79.5   |

--------------------------------------
## Confusion matrix of the validated modeling

  The confusion matrix presented herein contains the test IoU evaluation of the validated models (n=3035).
The rows of the confusion matrix represent the expected classes and the columns represent the predicted classes. 
The main diagonal of this matrix contains the IoU by class. The values of the confusion matrix are normalized by the
expected class (by row), where the total per class is the sum of its rows and columns.

#### Confusion matrix of model LigandRegion.

| K=1        | Background | Atom     |
|------------|------------|----------|
| Background | **86.3**   | 6.1      |
| Atom       | 17.4       | **68.4** |

#### Confusion matrix of model AtomCycle.

| k=1        | Background |    C     |   Atom   |
|------------|:----------:|:--------:|:--------:|
| Background |  **86.4**  |   1.7    |   3.2    |
| C          |    16.1    | **67.7** |   4.4    |
| Atom       |    23.4    |   1.7    | **58.8** |

#### Confusion matrix of model AtomC347CA56.

| k=13       | Background |   Atom   |    C5    |    CA5   |    C6    |    CA6   |    C3    |    C4    |    C7    |
|------------|:----------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| Background |  **86.5**  |    3.1   |    0.2   |    0.2   |    0.5   |    0.7   |    0.0   |    0.0   |    0.0   |
| Atom       |    24.1    | **58.9** |    0.4   |    0.1   |    0.9   |    0.9   |    0.0   |    0.0   |    0.0   |
| C5         |    14.5    |    2.9   | **63.4** |    1.3   |    2.2   |    1.3   |    0.0   |    0.0   |    0.0   |
| CA5        |    14.0    |    2.2   |    2.8   | **63.8** |    0.8   |    6.6   |    0.0   |    0.0   |    0.0   |
| C6         |    16.5    |    3.0   |    0.4   |    0.0   | **52.4** |    5.3   |    0.0   |    0.0   |    0.0   |
| CA6        |    13.6    |    2.4   |    0.3   |    0.4   |    3.4   | **62.0** |    0.0   |    0.0   |    0.0   |
| C3         |    20.3    |   21.3   |    0.0   |    0.0   |    3.0   |    0.2   | **26.3** |    1.0   |    0.0   |
| C4         |    25.0    |   12.2   |   11.3   |    2.8   |    0.1   |    3.9   |    0.0   | **26.6** |    0.0   |
| C7         |    17.4    |    2.7   |    0.8   |    0.2   |   18.9   |    5.9   |    0.6   |    0.0   | **42.1** |

#### Confusion matrix of model AtomSymbolGroups.

| k=1        | Background | C        | O        | N        | PSe      | Halo     |
|------------|------------|----------|----------|----------|----------|----------|
| Background |   **86.7** |      3.5 |      1.1 |      0.2 |      0.1 |      0.0 |
| C          |       17.9 | **60.4** |      1.3 |      0.7 |      0.1 |      0.0 |
| O          |       22.5 |      8.6 | **53.2** |      0.8 |      0.6 |      0.1 |
| N          |       15.1 |     17.9 |      3.2 | **51.2** |      0.1 |      0.1 |
| PSe        |        9.6 |      4.8 |      4.4 |      0.1 | **64.7** |      0.3 |
| Halo       |       21.6 |     13.1 |     11.1 |      0.2 |      1.0 | **37.6** |

--------------------------------------------

### Best setup of the training pipeline 

  The values presented below were optimized in the systematic analysis for model AtomC347CA56. 
This setup was used to train all the validated models and was defined as the default value of each respective parameter. 
The name of the parameter of the training pipeline used to define each setup is also presented.

|            **Setup**            |           **Parameter**            |              **Value**             |
|:-------------------------------:|:----------------------------------:|:----------------------------------:|
|       Deep neural network       |              --model               |    MinkUNet34C_CONVATROUS_HYBRID   |
|        Ligand image type        |             --pc_type              |             qRankMask_5            |
|            Optimizer            |            --optimizer             |                 SGD                |
|       Optimizer parameters      | --sgd_momentum and --sgd_dampening | momentum = 0.9 and dampening = 0.1 |
|          Learning rate          |                --lr                |                 2⁻⁸                |
|          Loss function          |            --loss_func             |                 wSL                |
|          Rotation rate          |          --rotation_rate           |                 50%                |
|         Total batch size        |     --batch_size and --num_gpu     |                 16                 |
| Number of gradient accumulation |            --iter_size             |                  1                 |
|        Normalization type       |              --model               |                 BN                 |


---------------------------------------------

## How to train a DL model

The training pipeline was implemented in the 'main.py' script. To see the list of parameters run:

`python main.py --help`

To see the list of mandatory parameters run:

`python main.py`

The following arguments are required:
> --ligs_data_filepath : This is the path to a training dataset containing the ligands entries to be used.

> --lig_pcdb_path : This is the path to a Lig-PCDB database.

> --vocab_path : This is the path to the vocabulary used to label the provided database. The 'class_mapping_path' parameters must be informed to used a mapped vocabulary.

The output directory is defined with the following parameter:
> --log_dir : The output logging directory will be named as: "<log_dir>\_<'train'|'test'>_<pc_type>\_kfold\_\<kfold>\_model-\<model>\_<current_time>" 

To train a DL model, first the number of threads for multiprocessing parallelization must be set using the variable 'OMP_NUM_THREADS', following [Minkwoski Engine](https://github.com/NVIDIA/MinkowskiEngine) setup (example with 8 threads).
Then, the pipeline may be executed passing the desired parameters. 
The parameter `--resume` may be used to continue the training of a previous trained model. 

Example training model AtomC347CA56, executed from this repository root folder:

```
conda activate np3_lig
export OMP_NUM_THREADS=8
python main.py --ligs_data_filepath training_dataset_valid_ligands_undersampling_maxLigCode_1000_kfolds_13_gridspace_0.5_SP.csv --lig_pcdb_path Lig-PCDB-SP/ --vocab_path ../np3_Lig-PCDB/vocabularies/SP-based/vocabulary_valid_ligands_PDB_1.5_2.2_SP-based.txt --class_mapping_path ../np3_Lig-PCDB/vocabularies/SP-based/mapping_atomC347CA56.csv --iter_size 1 --batch_size 8 --num_gpu 2 --gpu_index 0,1 --max_epoch 200 --log_dir output/train_lr-8_lossSL_w1-10-5-5-50-5-500C347_batch16iter1_gridsp0.5 --num_workers 8 --num_val_workers 4 --val_batch_size 4 --test_batch_size 4 --loss_weights 1,10,5,5,50,5,500,500,500 --kfold 1
```

------------------------------------------
## How to test a DL model

  The parameter `--is_train` controls if the training pipeline will train (True) or test (False) a model.
And the parameter `--weights` is used to load a previous trained model for testing.

Example of testing the model AtomC347CA56 with k=13:
```
export OMP_NUM_THREADS=8
python main.py --is_train False --ligs_data_filepath training_dataset_valid_ligands_undersampling_maxLigCode_1000_kfolds_13_gridspace_0.5_SP.csv --lig_pcdb_path Lig-PCDB-SP/ --vocab_path ../np3_Lig-PCDB/vocabularies/SP-based/vocabulary_valid_ligands_PDB_1.5_2.2_SP-based.txt --class_mapping_path ../np3_Lig-PCDB/vocabularies/SP-based/mapping_atomC347CA56.csv --log_dir output/test_lr-8_lossSL_w1-10-5-5-50-5-500C347_batch16iter1_gridsp0.5 --num_workers 8 --test_batch_size 4 --loss_weights 1 --kfold 13 --weights ../np3_blob_label/models/AtomC347CA56/modelAtomC347CA56_ligs-78911_img-qRankMask_5_gridspace-05_k13.ckpt
```

#### Test and save the predictions

  To save the predictions result of a testing, the parameters `--save_prediction` and `--save_pred_dir` must be defined together with `--test_batch_size 1`.

Example of testing a DL model and saving the predictions result.

```
python main.py --is_train False --ligs_data_filepath training_dataset_valid_ligands_undersampling_maxLigCode_1000_kfolds_13_gridspace_0.5_SP.csv --lig_pcdb_path Lig-PCDB-SP/ --vocab_path ../np3_Lig-PCDB/vocabularies/SP-based/vocabulary_valid_ligands_PDB_1.5_2.2_SP-based.txt --class_mapping_path ../np3_Lig-PCDB/vocabularies/SP-based/mapping_atomC347CA56.csv --log_dir output/test_lr-8_lossSL_w1-10-5-5-50-5-500C347_batch16iter1_gridsp0.5 --num_workers 8 --test_batch_size 1 --loss_weights 1 --kfold 13 --weights ../np3_blob_label/models/AtomC347CA56/modelAtomC347CA56_ligs-78911_img-qRankMask_5_gridspace-05_k13.ckpt --save_prediction True --save_pre_dir output/prediction_dir_test_lr-8_lossSL_w1-10-5-5-50-5-500C347_batch16iter1_gridsp0.5 
```
 

-----------------------------------------------

## How to visualize the training curves

The visualization of the training curves of a training job is done with the [TensorboadX](https://www.tensorflow.org/tensorboard) platform.

Example:
```
tensorboard --logdir=<your_log_dir>
```

-----------------------------------------------

## How to visualize the prediction results

The visualization of the predictions result, together with an error mask of each test entry, can be assessed with the following code:

```
python src/visualize_predictions.py output/prediction_dir_test_lr-8_lossSL_w1-10-5-5-50-5-500C347_batch16iter1_gridsp0.5
```

The error mask image have the points with a wrong prediction colored in red and the rest in grey. 
The points predicted as Background class are removed in another image to ease the visualization of the results.

----------------------------

## Available data


- The validated DL models are presented in the np3_blob_label repository, inside the folder named as 'models'. It contains 4 subfolders, one for each validated modeling containing:
  - The trained models in .ckpt format
  - A metadata table describing more information about the training setup of the available DL models

#### Lig-PCDB databases record

The databases created by Lig-PCDB and the validated models can be retrieved from [Zenodo](https://zenodo.org/), an open dissemination research data repository. The deposit data is located in the record xTODOlinkX, and contains:

- Lig-PCDB-SP_record : The database with the SP-based modeling images, vocabulary, structure labeling result (xyz record) and validated DL models.
- Lig-PCDB-AtomSymbol_record : The database with the AtomSymbol-based modeling images, vocabulary, structure labeling result (xyz record) and validated DL models.
- Lig-PCDB-Grids_reso-1.5-2.2_gridspace-0.5 : The database with the ligand grid image of the valid ligands list.


---------------------------------------------------------------

## Citing

_Paper in preparation to be published._
