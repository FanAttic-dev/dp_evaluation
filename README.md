# Autocam Evaluation

## Getting started

1. Initialize all submodules:
  
    ```shell
    git submodule update --init --recursive
    ```

2. Download the pretrained segmentation for TVCalib:

    ```shell
    mkdir -p assets/weights/segment_localization
    wget https://tib.eu/cloud/s/x68XnTcZmsY4Jpg/download/train_59.pt -O assets/weights/segment_localization/train_59.pt
    ```

3. Create a [Conda environment](https://docs.conda.io/en/latest/) by first installing, e.g., [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) and then running these commands:

    ```shell
    conda env create -f environment.yml
    conda activate tvcalib
    ```

4. 