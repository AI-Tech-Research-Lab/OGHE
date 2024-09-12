## OGHE - Oncological Genomic analysis over HE

The repository contains the code needed to replicate the experiments presented in "Enhancing privacy-preserving cancer classification with Convolutional Neural Networks". Dataset is available at https://drive.google.com/drive/folders/16taBYFtPdygWOZAS60Y6L3z9Z-YGSoIq?usp=sharing .

In particular:
- `.devcontainer` contains the files needed to create a devcontainer using VSCode;
- `Dataset` contains the iDash2020 dataset;
- `requirements.txt` is the Python requirements file. Install all the required libraries with `pip install -r requirements.txt`;
- `data_utils.py` contains the pre-processing functions, needed to parse and load the dataset;
- `experiment_*_shuffle.py` contains the code of the FC, NN, and OGHE with the data pre-processing of Hong et al.;
- `experiment_*_shuffle_OurPreProcessing.py` contains the code of the FC, NN, and GenHECNN with our proposed pre-processing;
- `StatisticalTest` contains the code to run the statistical tests of difference between the proposed solution and Hong et al.;
- `Timing` contains the code for the encrypted computation time. In particular, to run the timing experiments, run:

## Encrypted experiments
To run the encrypted tests, run
```
python3 Test_single.py --n_jobs=1 --n_samples=1
```
for a single sample; note that it will also output the final content of the ciphertexts to check the correctness vs the plain processing, and

```
python3 Test_multithread.py --n_jobs 40 --n_samples 100
```
for multiple samples.
