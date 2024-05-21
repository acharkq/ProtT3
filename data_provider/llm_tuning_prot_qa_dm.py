# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from data_provider.prot_qa_dm import PDBQADataset
from data_provider.gal_helpers import escape_custom_split_sequence


class LLMTuningProtQACollater(object):
    def __init__(self, tokenizer, q_max_len, a_max_len, use_gal, prompt):
        self.tokenizer = tokenizer
        self.q_max_len = q_max_len
        self.a_max_len = a_max_len
        self.use_gal = use_gal
        self.prompt = prompt
        assert prompt.find('{}') >= 0
        
    def __call__(self, batch):
        prot_seqs, questions, answers, q_types = zip(*batch)
        assert len(prot_seqs) == len(questions) == len(answers)
        questions = [self.prompt.format(prot_seqs[i], questions[i]) for i in range(len(prot_seqs))]

        if self.use_gal:
            questions = [escape_custom_split_sequence(q) for q in questions]
        answers = [a + '\n' for a in answers]
        if False:
            self.tokenizer.padding_side = 'left'
            q_batch = self.tokenizer(questions,
                                    truncation=True,
                                    padding='max_length',
                                    add_special_tokens=True,
                                    max_length=self.q_max_len,
                                    return_tensors='pt',
                                    return_attention_mask=True, 
                                    return_token_type_ids=False)
            self.tokenizer.padding_side = 'right'
            a_batch = self.tokenizer(answers,
                                    truncation=True,
                                    padding='max_length',
                                    add_special_tokens=True,
                                    max_length=self.a_max_len,
                                    return_tensors='pt',
                                    return_attention_mask=True, 
                                    return_token_type_ids=False)
            return q_batch, a_batch
        else:
            self.tokenizer.padding_side = 'right'
            qa_pair = [[q, a] for q, a in zip(questions, answers)]
            qa_batch = self.tokenizer(qa_pair,
                                      truncation=True,
                                      padding='max_length',
                                      add_special_tokens=True,
                                      max_length=self.q_max_len + self.a_max_len,
                                      return_tensors='pt',
                                      return_attention_mask=True,
                                      return_token_type_ids=True)
            return qa_batch


class InferenceCollater(object):
    def __init__(self, tokenizer, q_max_len, a_max_len, use_gal, prompt):
        self.tokenizer = tokenizer
        self.q_max_len = q_max_len
        self.a_max_len = a_max_len
        self.use_gal = use_gal
        self.prompt = prompt
        assert prompt.find('{}') >= 0
        
    def __call__(self, batch):
        prot_seqs, questions, answers, q_types, indices = zip(*batch)
        assert len(prot_seqs) == len(questions) == len(answers)
        questions = [self.prompt.format(prot_seqs[i], questions[i]) for i in range(len(prot_seqs))]

        if self.use_gal:
            questions = [escape_custom_split_sequence(q) for q in questions]
        answers = [a + '\n' for a in answers]
        self.tokenizer.padding_side = 'left'
        q_batch = self.tokenizer(questions,
                                 truncation=True,
                                 padding='max_length',
                                 add_special_tokens=True,
                                 max_length=self.q_max_len,
                                 return_tensors='pt',
                                 return_attention_mask=True, 
                                 return_token_type_ids=False)
        target_dict = {'targets': answers, 'q_types': q_types, 'indices': indices}
        return q_batch, target_dict


class LLMTuningProtQADM(LightningDataModule):
    def __init__(
        self,
        root: str = 'data/',
        args=None,
    ):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = args.num_workers
        self.q_max_len = args.q_max_len
        self.a_max_len = args.a_max_len
        self.prompt = args.prompt
        
        self.train_dataset = PDBQADataset(root, 'train.txt', "Question: {} Answer:", filter_side_qa=args.filter_side_qa)
        self.val_dataset = PDBQADataset(root, 'val.txt', "Question: {} Answer:", filter_side_qa=args.filter_side_qa)
        self.test_dataset = PDBQADataset(root, 'test.txt', "Question: {} Answer:", filter_side_qa=args.filter_side_qa)
        
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
            collate_fn=LLMTuningProtQACollater(self.tokenizer, self.q_max_len, self.a_max_len, self.use_gal, self.prompt),
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
            collate_fn=LLMTuningProtQACollater(self.tokenizer, self.q_max_len, self.a_max_len, self.use_gal, self.prompt),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=InferenceCollater(self.tokenizer, self.q_max_len, self.a_max_len, self.use_gal, self.prompt),
        )
        return [val_loader, test_loader]
    

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--root', type=str, default='data/SwissProtV3')
        parser.add_argument('--q_max_len', type=int, default=1064)
        parser.add_argument('--a_max_len', type=int, default=36)
        parser.add_argument('--prompt', type=str, default='[START_AMINO]{}[END_AMINO]. {}')
        parser.add_argument('--filter_side_qa', action='store_true', default=False)
        return parent_parser


