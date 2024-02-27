import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, TrainingArguments
from datasets import load_from_disk, concatenate_datasets, Dataset
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, NoReturn, Union
from contextlib import contextmanager
import time
import json
import argparse

from dpr_encoder import DPREncoder

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class DenseRetrieval:

    def __init__(self, args, dataset, num_neg, tokenizer, p_encoder, q_encoder):

        '''
        학습과 추론에 사용될 여러 셋업을 마쳐봅시다.
        '''

        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_path = "../data"

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        # wiki로 사용할 데이터셋 경로 추가
        self.wiki_path = os.path.join(self.data_path, "wikipedia_documents.json")

        self.prepare_in_batch_negative(num_neg=num_neg)

    def get_dense_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name_q_encoder = f"dense_embedding_q_encoder.bin"
        pickle_name_p_encoder = f"dense_embedding_p_encoder.bin"
        q_emd_path = os.path.join(self.data_path, pickle_name_q_encoder)
        p_emd_path = os.path.join(self.data_path, pickle_name_p_encoder)

        if os.path.isfile(q_emd_path) and os.path.isfile(p_emd_path):
            with open(q_emd_path, "rb") as file:
                self.p_encoder = pickle.load(file)
            with open(p_emd_path, "rb") as file:
                self.q_encoder = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.train()
            with open(q_emd_path, "wb") as file:
                pickle.dump(self.p_encoder, file)
            with open(p_emd_path, "wb") as file:
                pickle.dump(self.q_encoder, file)
            print("Embedding pickle saved.")

    def prepare_in_batch_negative(self, dataset=None, num_neg=2, tokenizer=None):

        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.
        corpus = np.array(list(set([example for example in dataset['context']])))
        p_with_neg = []

        for c in dataset['context']:

            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size)

        valid_seqs = tokenizer(dataset['context'], padding="max_length", truncation=True, return_tensors='pt')
        passage_dataset = TensorDataset(
            valid_seqs['input_ids'], valid_seqs['attention_mask'], valid_seqs['token_type_ids']
        )
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size)


    def train(self, args=None):

        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in range(int(args.num_train_epochs)):
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.p_encoder.train()
                    self.q_encoder.train()

                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        'input_ids': batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'attention_mask': batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'token_type_ids': batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }

                    q_inputs = {
                        'input_ids': batch[3].to(args.device),
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }

                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f'{str(loss.item())}')

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

    def retrieve(
        self, dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        
        assert self.p_encoder is not None and self.q_encoder is not None, "get_dense_embedding() 메소드를 먼저 수행해줘야합니다."

        # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
        total = []
        doc_scores, doc_indices = self.get_relevant_doc_bulk(
            dataset["question"], k=topk
        )
        for idx, example in enumerate(
            tqdm(dataset, desc="Dense retrieval: ")
        ):
            tmp = {
                # Query와 해당 id를 반환합니다.
                "question": example["question"],
                "id": example["id"],
                # Retrieve한 Passage의 id, context를 반환합니다.
                "context": " ".join(
                    # [self.contexts[i] for i in doc_indices[idx]]
                    [self.dataset[i]["context"] for i in doc_indices[idx]]
                ),
            }
            if "context" in example.keys() and "answers" in example.keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)

        cqas = pd.DataFrame(total)
        return cqas
    
    def get_wiki_dataloader(self, wiki_path):
        with open(wiki_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )

        wiki_seqs = self.tokenizer(self.contexts, padding="max_length", truncation=True, return_tensors='pt')
        wiki_dataset = TensorDataset(
            wiki_seqs['input_ids'], wiki_seqs['attention_mask'], wiki_seqs['token_type_ids']
        )
        wiki_dataloader = DataLoader(wiki_dataset, batch_size=self.args.per_device_train_batch_size)
        return wiki_dataloader
    
    def get_p_embedding(self, p_encoder):
        # p_dataloader = self.get_wiki_dataloader(self.wiki_path)
        p_dataloader = self.passage_dataloader
        with torch.no_grad():
            p_encoder.eval()

            p_embs = []
            for batch in tqdm(p_dataloader):
                p_inputs = {
                    'input_ids': batch[0].to(self.device),
                    'attention_mask': batch[1].to(self.device),
                    'token_type_ids': batch[2].to(self.device)
                }
                p_emb = p_encoder(**p_inputs).to('cpu')
                p_embs.append(p_emb)
        p_embs = torch.cat(p_embs, dim=0)
        return p_embs
    
    def get_q_embedding(self, q_encoder, queries):
        q_seqs = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt').to(self.device)
        q_dataset = TensorDataset(
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )
        q_dataloader = DataLoader(q_dataset, batch_size=self.args.per_device_train_batch_size)
        with torch.no_grad():
            q_encoder.eval()
            
            q_embs = []
            for batch in tqdm(q_dataloader):
                q_inputs = {
                    'input_ids': batch[0].to(self.device),
                    'attention_mask': batch[1].to(self.device),
                    'token_type_ids': batch[2].to(self.device)
                }
                q_emb = q_encoder(**q_inputs).to('cpu')
                q_embs.append(q_emb)

        q_embs = torch.cat(q_embs, dim=0)
        return q_embs
    
    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        print("q_embedding start")
        q_embs = self.get_q_embedding(self.q_encoder, queries)
        print("p_embedding start")
        p_embs = self.get_p_embedding(self.p_encoder)

        print("embedding done")

        result = torch.matmul(q_embs, torch.transpose(p_embs, 0, 1))

        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = torch.argsort(result[i, :], dim=-1, descending=True).squeeze()
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Argparse')

    parser.add_argument('--output_dir', type=str, default="dense_retireval")
    parser.add_argument('--evaluation_strategy', type=str, default="epoch")
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2)
    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--model_name', type=str, default="klue/bert-base")
    parser.add_argument('--num_neg', type=int, default=2)
    parser.add_argument('--data_path', type=str, default="../data/train_dataset")

    args = parser.parse_args()

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    p_encoder = DPREncoder.from_pretrained(args.model_name).to(device)
    q_encoder = DPREncoder.from_pretrained(args.model_name).to(device)

    # data_path 입력
    org_dataset = load_from_disk(args.data_path)
    dataset = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(dataset)

    retriever = DenseRetrieval(train_args, dataset, args.num_neg, tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder)
    retriever.get_dense_embedding()

    # result = retriever.retrieve(dataset, topk=1)
    # result.to_csv("./outputs/result.csv", index=False)

