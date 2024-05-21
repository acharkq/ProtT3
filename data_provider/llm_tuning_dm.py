# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
from data_provider.gal_helpers import escape_custom_split_sequence
from data_provider.stage1_dm import SwissProtDataset, OntoProteinDataset
from torch.utils.data import DataLoader, ConcatDataset


class LLMTuningCollater:
    def __init__(self, tokenizer, text_max_len, prot_max_len, use_gal):
        self.text_max_len = text_max_len
        self.prot_max_len = prot_max_len
        self.tokenizer = tokenizer
        self.use_gal = use_gal
        
    def __call__(self, batch):
        prot_seqs, prompt_seqs, text_seqs, _ = zip(*batch)
        prot_seqs = [prompt.format(p) for prompt, p in zip(prompt_seqs, prot_seqs)]
        if self.use_gal:
            prot_seqs = [escape_custom_split_sequence(p) for p in prot_seqs]
        ## deal with prompt
        self.tokenizer.padding_side = 'left'
        prot_batch = self.tokenizer(text=prot_seqs,
                                    truncation=True,
                                    padding='max_length',
                                    add_special_tokens=True,
                                    max_length=self.prot_max_len,
                                    return_tensors='pt',
                                    return_attention_mask=True)
        self.tokenizer.padding_side = 'right'
        text_batch = self.tokenizer(text=text_seqs,
                                    truncation=True,
                                    padding='max_length',
                                    add_special_tokens=True,
                                    max_length=self.text_max_len,
                                    return_tensors='pt',
                                    return_attention_mask=True)
        return prot_batch, text_batch


class InferenceCollater:
    def __init__(self, tokenizer, text_max_len, prot_max_len, use_gal):
        self.text_max_len = text_max_len
        self.prot_max_len = prot_max_len
        self.tokenizer = tokenizer
        self.use_gal = use_gal
        
    def __call__(self, batch):
        prot_seqs, prompt_seqs, text_seqs, indices = zip(*batch)
        prot_seqs = [prompt.format(p) for prompt, p in zip(prompt_seqs, prot_seqs)]
        if self.use_gal:
            prot_seqs = [escape_custom_split_sequence(p) for p in prot_seqs]
        ## deal with prompt
        self.tokenizer.padding_side = 'left'
        prot_batch = self.tokenizer(text=prot_seqs,
                                    truncation=True,
                                    padding='max_length',
                                    add_special_tokens=True,
                                    max_length=self.prot_max_len,
                                    return_tensors='pt',
                                    return_attention_mask=True)
        target_dict = {'targets': text_seqs, 'indices': indices}
        return prot_batch, target_dict



class LLMTuningDM(LightningDataModule):
    def __init__(
        self,
        root: str = 'data/',
        args=None,
    ):
        super().__init__()
        self.batch_size = args.batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = args.num_workers
        self.prot_max_len = args.prot_max_len
        self.text_max_len = args.text_max_len
        if root.find('SwissProtV3') >= 0:
            self.train_dataset = SwissProtDataset(root+'/train_set.json', prompt='[START_AMINO]{}[END_AMINO]. Swiss-Prot description: ', return_prompt=True)
            self.val_dataset = SwissProtDataset(root+'/valid_set.json', prompt='[START_AMINO]{}[END_AMINO]. Swiss-Prot description: ', return_prompt=True)
            self.test_dataset = SwissProtDataset(root+'/test_set.json', prompt='[START_AMINO]{}[END_AMINO]. Swiss-Prot description: ', return_prompt=True)
        elif root.find('OntoProteinDatasetV2') >= 0:
            self.train_dataset = OntoProteinDataset(root+'/train.txt', prompt='[START_AMINO]{}[END_AMINO]. Gene Ontology description: ', return_prompt=True)
            self.val_dataset = OntoProteinDataset(root+'/valid.txt', prompt='[START_AMINO]{}[END_AMINO]. Gene Ontology description: ', return_prompt=True)
            self.test_dataset = OntoProteinDataset(root+'/test.txt', prompt='[START_AMINO]{}[END_AMINO]. Gene Ontology description: ', return_prompt=True)
        else:
            raise NotImplementedError()
        self.tokenizer = None
        self.use_gal = args.llm_name.find('gal') >= 0
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False,
            collate_fn=LLMTuningCollater(self.tokenizer, self.text_max_len, self.prot_max_len, self.use_gal),
        )
        return loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=LLMTuningCollater(self.tokenizer, self.text_max_len, self.prot_max_len, self.use_gal),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.prot_max_len, self.use_gal),
        )
        return [val_loader, test_loader]

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--root', type=str, default='data/SwissProtV3')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prot_max_len', type=int, default=1024)
        parser.add_argument('--q_max_len', type=int, default=1064)
        parser.add_argument('--a_max_len', type=int, default=36)
        parser.add_argument('--prompt', type=str, default='[START_AMINO]{}[END_AMINO]. The protein has the following properties: ')
        parser.add_argument('--filter_side_qa', action='store_true', default=False)
        return parent_parser


class LLMTuningMixDM(LightningDataModule):
    def __init__(
        self,
        root: str = 'data/',
        args=None,
    ):
        super().__init__()
        self.batch_size = args.batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = args.num_workers
        self.prot_max_len = args.prot_max_len
        self.text_max_len = args.text_max_len
        
        train_dataset1 = SwissProtDataset(root+'/SwissProtV3/train_set.json', prompt='[START_AMINO]{}[END_AMINO]. Swiss-Prot description: ', return_prompt=True)
        train_dataset2 = OntoProteinDataset(root+'/OntoProteinDatasetV2/train.txt', prompt='[START_AMINO]{}[END_AMINO]. Gene Ontology description: ', return_prompt=True)
        self.train_dataset = ConcatDataset([train_dataset1, train_dataset2])
        self.swiss_val_dataset = SwissProtDataset(root+'/SwissProtV3/valid_set.json', prompt='[START_AMINO]{}[END_AMINO]. Swiss-Prot description: ', return_prompt=True)
        self.onto_val_dataset = OntoProteinDataset(root+'/OntoProteinDatasetV2/valid.txt', prompt='[START_AMINO]{}[END_AMINO]. Gene Ontology description: ', return_prompt=True)
        self.swiss_test_dataset = SwissProtDataset(root+'/SwissProtV3/test_set.json', prompt='[START_AMINO]{}[END_AMINO]. Swiss-Prot description: ', return_prompt=True)
        self.onto_test_dataset = OntoProteinDataset(root+'/OntoProteinDatasetV2/test.txt', prompt='[START_AMINO]{}[END_AMINO]. Gene Ontology description: ', return_prompt=True)
        
        self.tokenizer = None
        self.use_gal = args.llm_name.find('gal') >= 0
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False,
            collate_fn=LLMTuningCollater(self.tokenizer, self.text_max_len, self.prot_max_len, self.use_gal),
        )
        return loader
    
    def val_dataloader(self):
        swiss_val_loader = DataLoader(
            self.swiss_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=LLMTuningCollater(self.tokenizer, self.text_max_len, self.prot_max_len, self.use_gal),
        )
        swiss_test_loader = DataLoader(
            self.swiss_test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.prot_max_len, self.use_gal),
        )

        onto_val_loader = DataLoader(
            self.onto_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=LLMTuningCollater(self.tokenizer, self.text_max_len, self.prot_max_len, self.use_gal),
        )
        onto_test_loader = DataLoader(
            self.onto_test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.prot_max_len, self.use_gal),
        )
        return [swiss_val_loader, swiss_test_loader, onto_val_loader, onto_test_loader]

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--root', type=str, default='data/SwissProtV3')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prot_max_len', type=int, default=1024)
        parser.add_argument('--q_max_len', type=int, default=1064)
        parser.add_argument('--a_max_len', type=int, default=36)
        parser.add_argument('--prompt', type=str, default='[START_AMINO]{}[END_AMINO]. The protein has the following properties: ')
        parser.add_argument('--filter_side_qa', action='store_true', default=False)
        return parent_parser
    

if __name__ == '__main__':
    dataset = SwissProtDataset('../data/SwissProtV3/train_set.json')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('facebook/galactica-1.3b')
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False,
            collate_fn=LLMTuningCollater(tokenizer, 128, 1024, True, '[START_AMINO]{}[END_AMINO].'),
        )
    for data in loader:
        input()