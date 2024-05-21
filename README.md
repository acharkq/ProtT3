# ProtT3: Protein-to-Text Generation for Text-based Protein Understanding

Codes of our ACL2024 paper.

Authors: Zhiyuan Liu, An Zhang, Hao Fei, Enzhi Zhang, Xiang Wang, Kenji Kawaguchi, Tat-Seng Chua


## Dependencies

python==3.8

* Install PyTorch with cuda-11.7 using conda by following the instructions in [link](https://pytorch.org/get-started/locally/)
* Install flash-attention by running `pip install flash-attn --no-build-isolation`. You might need to install the following dependencies first, for building the flash-attention module:
    * `pip install packaging ninja`
    * `conda install -c "nvidia/label/cuda-11.7.1" cuda-nvcc`
    * `conda install -c "nvidia/label/cuda-11.7.1" cuda-libraries-dev`
* Install the lastest version of opendela by runing `pip install git+https://github.com/thunlp/OpenDelta.git`
* Install Lavis: `pip install rouge_score nltk salesforce-lavis`
* Install others: `pip install -U transformers pytorch-lightning`
* Install the lastest version of deepspeed: `pip install git+https://github.com/microsoft/DeepSpeed.git`
* Download nltk corpus:
```
import nltk
nltk.download('wordnet')
```

## Dataset

Download our pre-processed datasets from [link](https://osf.io/23azs/?view_only=185575515e714f4798499bf06513a730), and unzip the datasets under the `./data` directory

## Reproduce results by training from scratch

* Reproduce results in stage 1:

```sh
python stage1.py --devices '0,1,2,3' --mode train --filename stage1_ckpt --num_query_token 8 --plm_name "facebook/esm2_t30_150M_UR50D" --save_every_n_epochs 10 --batch_size 32 --precision 'bf16-mixed' --num_workers 8
```

* Convert stage1's DeepSpeed checkpoint to PyTorch format by running

```sh
python convert.py --input /path/to/stage1/ckpt/address --output /path/to/ckpt/saving/address
```

* Reproduce results in stage 2:

    * Protein Captioning:

        ```sh
        python stage2.py --devices '0,1,2,3' --mode train --filename protein_captioning_swiss_dataset --num_query_token 8  --save_every_n_epochs 10 --batch_size 32 --precision 'bf16-mixed' --num_workers 8 --llm_tune mid_lora --enable_flash --root './data/SwissProtV3' --stage1_path /path/to/ckpt/saving/address;
        ```

    * Protein Question-Answering:

        ```sh
        python stage2.py --devices '0,1,2,3' --mode train  --filename prot_qa --num_query_token 8  --save_every_n_epochs 10 --num_workers 8 --batch_size 128 --accumulate_grad_batches 1 --precision 'bf16-mixed'  --root "data/PDBDataset" --llm_tune mid_lora --prompt "Question: {} Answer:" --inference_batch 32 --max_inference_len 36  --stage1_path /path/to/ckpt/saving/address;
        ```

    * After running one of the two scripts above, the model's protein-to-text generation resuults will be saved at `./all_checkpoint/[filename]/lightning_logs/[version_x]/dataset0_predictions.txt`. You can evaluate the results by running

        ```sh 
        ## for question-answering evaluation
        python read_results --path ./all_checkpoint/[filename]/lightning_logs/[version_x]/dataset0_predictions.txt --qa_question 
        
        ## for protein captioning evaluation
        python read_results --path ./all_checkpoint/[filename]/lightning_logs/[version_x]/dataset0_predictions.txt 
        ```

## Reproduce results by loading our checkpoints

Download our released checkpoints from [link](https://osf.io/23azs/?view_only=185575515e714f4798499bf06513a730)

* Reproduce results in stage 1:

```sh
python stage1.py --devices '0,1,2,3' --mode eval --filename stage1_ckpt --num_query_token 8 --plm_name "facebook/esm2_t30_150M_UR50D" --save_every_n_epochs 10 --batch_size 32 --precision 'bf16-mixed' --num_workers 8 --init_checkpoint /path/to/stage1.ckpt;
```

* Reproduce results in stage 2:

    * Protein Captioning:

        ```sh
        python stage2.py --devices '0,1,2,3' --mode train --filename protein_captioning_swiss_dataset --num_query_token 8  --save_every_n_epochs 10 --batch_size 32 --precision 'bf16-mixed' --num_workers 8 --llm_tune mid_lora --enable_flash --root './data/SwissProtV3' --init_checkpoint /path/to/swiss_ft.ckpt;
        ```

    * Protein Question-Answering:

        ```sh
        python stage2.py --devices '0,1,2,3' --mode train  --filename prot_qa --num_query_token 8  --save_every_n_epochs 10 --num_workers 8 --batch_size 128 --accumulate_grad_batches 1 --precision 'bf16-mixed'  --root "data/PDBDataset" --llm_tune mid_lora --prompt "Question: {} Answer:" --inference_batch 32 --max_inference_len 36  --init_checkpoint /path/to/pdbqa_ft.ckpt;
        ```


## Citation

```bib
@inproceedings{liu2024prott,
    title={ProtT3: Protein-to-Text Generation for Text-based Protein Understanding},
    author={Liu, Zhiyuan and Zhang, An and Fei, Hao and Zhang, Enzhi and Wang, Xiang and Kawaguchi, Kenji and Chua, Tat-Seng},
    booktitle={{ACL}},
    publisher    = {Association for Computational Linguistics},
    year={2024},
    url={https://openreview.net/forum?id=ZmIjOPil2b}
}
```
