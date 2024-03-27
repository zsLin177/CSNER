from supar.utils.logging import init_logger, logger, progress_bar
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import os
from datetime import datetime, timedelta
from supar.models.seqtag import CrfSeqTagModel, SentLabelModel
from dataset import Dataset, collate_fn, read_json, InsDataset, ins_collate_fn
import json
from supar.utils.metric import Metric, NERMetric, NoPosNERMetric, LevenNERMetric, LevenMetric
import torch.nn as nn
from transformers import AutoTokenizer
import gc


class Parser(object):
    def __init__(self, args):
        self.args = args
        self.labels_num = 7

    def load_model(self, model_path):
        state = torch.load(model_path)
        self.model.load_state_dict(state['state_dict'], False)
    
    @torch.no_grad()
    def select_data(self, iter: int, best_iter = None):
        """
        use the current model to select the unlabeled data
        """
        # load last best model
        s_model = CrfSeqTagModel(n_words=AutoTokenizer.from_pretrained(self.args.roberta_path).vocab_size, encoder="bert", n_mlp=800, n_labels=self.labels_num, bert=self.args.roberta_path)
        if self.args.device != '-1':
            s_model = s_model.to(torch.device("cuda"))
        if iter > 1:
            model_path = os.path.join(self.args.path, f"iter-{best_iter}-best.model")
        else:
            model_path = self.args.init_model_path
        s_model.load_state_dict(torch.load(model_path)["state_dict"])
        s_model.eval()

        init_logger(logger)
        dataset = Dataset(
            file_name=self.args.unlabel_file,
            tokenizer_path=self.args.roberta_path,
            astrain=False,
            icsr=self.args.icsr,
        )
        dataloader = DataLoader(dataset,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(
                                    SequentialSampler(dataset),
                                    batch_size=self.args.batch_size,
                                    drop_last=False,
                                ),
                                num_workers=self.args.n_workers)
        logger.info(f"\n all unlabel dataset: {len(dataset):6}\n")
        logger.info(f"Selecting the data for self-training from : {self.args.unlabel_file} with model {model_path}")

        bar = progress_bar(dataloader)
        sum_selected = 0
        if not os.path.exists(self.args.select_file_dir):
            os.mkdir(self.args.select_file_dir)

        with open(os.path.join(self.args.select_file_dir, f"iter-{iter}.jsonl"), 'w', encoding="utf8") as f:
            # write the src data
            src_data = read_json(self.args.src_train_file)
            for dic in src_data:
                f.write(json.dumps(dic, ensure_ascii=False) + "\n")
            for batch in bar:
                keys = batch["keys"]
                sents = batch["parse_sents"]
                words = batch["tokenids"]
                if self.args.device != '-1':
                    words = words.to(torch.device("cuda"))

                word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
                mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
                score = s_model(words)
                output, output_p = s_model.decode(score, mask)
                mask = mask[:, 1:]
                output = output.masked_fill(~mask, dataset.label_stoi["O"])

                # select the data for self-training based on probability of the predicted label sequence
                # for the all O label sequence and the output_p is 1, we drop it with probability p_drop
                # [batch_size]
                if self.args.dynamic_p_hold:
                    p_hold = self.args.p_hold + ((1-self.args.p_hold) / self.args.iter) * (iter-1)
                else:
                    p_hold = self.args.p_hold
                p_large_p_hold = output_p.ge(p_hold)
                all_o = output.eq(dataset.label_stoi["O"]).all(-1)
                random_p = torch.rand_like(all_o, dtype=torch.float)
                allo_select_mask = p_large_p_hold & all_o & (random_p > self.args.p_drop)
                other_select_mask = p_large_p_hold & (~all_o)
                select_mask = allo_select_mask | other_select_mask
                sum_selected += select_mask.sum().item()

                nes = self.get_nes(output, sents, dataset.label_itos)
                for key, ne, sent, select in zip(keys, nes, sents, select_mask):
                    if select:
                        this_dic = {"key": key, "text": sent, "label": ne}
                        f.write(json.dumps(this_dic, ensure_ascii=False) + "\n")
                        f.flush()

        logger.info(f"Selected {sum_selected} examples, ratio: {sum_selected / len(dataset):.4f}, combined with src train {self.args.src_train_file}, saved to {os.path.join(self.args.select_file_dir, f'iter-{iter}.jsonl')}.")
        with open(os.path.join(self.args.select_file_dir, f"iter-{iter}.finish"), 'w', encoding="utf8") as f:
            f.write(f"{iter} data selecting finished.")
        # del s_model
        s_model = s_model.cpu()
        del s_model
        torch.cuda.empty_cache()
        gc.collect()

    def self_training(self):
        init_logger(logger)

        dev_set = Dataset(
            file_name=self.args.dev_file,
            tokenizer_path=self.args.roberta_path,
            astrain=True,
            icsr=self.args.icsr,
        )
        dev_loader = DataLoader(dev_set,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(
                                    SequentialSampler(dev_set),
                                    batch_size=self.args.batch_size,
                                    drop_last=False,
                                ),
                                num_workers=self.args.n_workers)
        logger.info(f"\n dev_set: {len(dev_set):6}\n")
        test_set = Dataset(
            file_name=self.args.test_file,
            tokenizer_path=self.args.roberta_path,
            astrain=True,
            icsr=self.args.icsr,
        )
        test_loader = DataLoader(test_set,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(
                                    SequentialSampler(test_set),
                                    batch_size=self.args.batch_size,
                                    drop_last=False,
                                ),
                                num_workers=self.args.n_workers)
        logger.info(f"\n test_set: {len(test_set):6}\n")

        best_iter, best_iter_metric = 1, Metric()

        if self.args.use_kl:
            # load the init model
            self.base_model = CrfSeqTagModel(n_words=AutoTokenizer.from_pretrained(self.args.roberta_path).vocab_size, encoder="bert", n_mlp=800, n_labels=self.labels_num, bert=self.args.roberta_path)
            if self.args.device != '-1':
                self.base_model = self.base_model.to(torch.device("cuda"))
            self.base_model.load_state_dict(torch.load(self.args.init_model_path)["state_dict"])
            self.base_model.eval()

        for iter in range(1, self.args.iter+1):
            # first use the last best model to select the unlabeled data
            logger.info(f"\nself training iteration: {iter} start")
            if os.path.exists(os.path.join(self.args.select_file_dir, f"iter-{iter}.finish")):
                logger.info(f"iteration: {iter} data has been selected")
            else:
                self.select_data(iter, best_iter)
            train_file = os.path.join(self.args.select_file_dir, f"iter-{iter}.jsonl")
            train_set = Dataset(train_file, self.args.roberta_path, astrain=True, icsr=self.args.icsr)
            train_loader = DataLoader(train_set,
                                    collate_fn=collate_fn,
                                    batch_sampler=BatchSampler(
                                        RandomSampler(train_set),
                                        batch_size=self.args.batch_size,
                                        drop_last=False,
                                    ),
                                    num_workers=self.args.n_workers)
            logger.info(f"\n train_set: {len(train_set):6}\n")

            # from stratch
            self.model = CrfSeqTagModel(n_words=AutoTokenizer.from_pretrained(self.args.roberta_path).vocab_size, encoder="bert", n_mlp=800, n_labels=self.labels_num, bert=self.args.roberta_path)
            if self.args.device != '-1':
                self.model = self.model.to(torch.device("cuda"))

            steps = (len(train_set)//self.args.batch_size) * self.args.epochs // self.args.update_steps
            optimizer = AdamW(
                [{'params': c, 'lr': self.args.lr * (1 if n.startswith('encoder') else self.args.lr_rate)}
                 for n, c in self.model.named_parameters()],
                self.args.lr)
            scheduler = get_linear_schedule_with_warmup(optimizer, int(steps*self.args.warmup), steps)

            elapsed = timedelta()
            best_e, best_metric = 1, Metric()
            for epoch in range(1, self.args.epochs+1):
                start = datetime.now()
                logger.info(f"Epoch {epoch} / {self.args.epochs}:")
                self._train(train_loader, optimizer, scheduler, train_set.label_itos)
                dev_metric = self._evaluate(dev_loader, dev_set.label_itos)
                logger.info(f" dev: - {dev_metric}")
                test_metric = self._evaluate(test_loader, test_set.label_itos)
                logger.info(f"test: - {test_metric}")
                t = datetime.now() - start
                elapsed += t
                if dev_metric > best_metric:
                    best_e, best_metric = epoch, dev_metric
                    state = {'state_dict': self.model.state_dict()}
                    torch.save(state, os.path.join(self.args.path, f"iter-{iter}-best.model"))
                    logger.info(f"{t}s elapsed, {elapsed}s in total, saved\n")
                else:
                    logger.info(f"{t}s elapsed, {elapsed}s in total\n")
                if epoch - best_e >= self.args.patience:
                    break
            logger.info(f"iteration: {iter} end")
            logger.info(f"Epoch {best_e} saved")
            logger.info(f"{'dev:':5} {best_metric}")

            if best_metric > best_iter_metric:
                best_iter, best_iter_metric = iter, best_metric
            if iter - best_iter >= self.args.patience:
                break
        logger.info(f"get best dev result at iteration {best_iter}\n")
        logger.info(f"{'dev:':5} {best_iter_metric}")

    def _train(self, dataloader, optimizer, scheduler, itos, model=None):
        stoi = {s: i for i, s in enumerate(itos)}
        if model is None:
            self.model.train()
            metric = NERMetric(itos)
            bar = progress_bar(dataloader)
            for i, batch in enumerate(bar, 1):
                words = batch["tokenids"]
                labels = batch["labelids"]
                if self.args.device != '-1':
                    words = words.to(torch.device("cuda"))
                    labels = labels.to(torch.device("cuda"))
                word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
                mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
                score = self.model(words)
                crf_loss = self.model.loss(score, labels, mask)
                if self.args.use_kl:
                    # use the kl divergence to regularize the model
                    s_score = self.base_model(words)
                    s_dist = self.base_model.dist(s_score, mask)
                    kl_loss = self.model.kl_loss(score, mask, s_dist)
                
                loss = crf_loss + self.args.kl_weight * kl_loss if self.args.use_kl else crf_loss
                loss = loss / self.args.update_steps
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                if i % self.args.update_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                preds, _ = self.model.decode(score, mask)
                mask = mask[:, 1:]
                metric(preds.masked_fill(~mask, stoi["O"]), labels)
                if not self.args.use_kl:
                    bar.set_postfix_str(f"lr: {scheduler.get_last_lr()[0]:.4e} - loss: {loss:.2f} - {metric}")
                else:
                    bar.set_postfix_str(f"lr: {scheduler.get_last_lr()[0]:.4e} - crf loss: {crf_loss:.2f} - kl loss: {kl_loss:.2f} - {metric}")
            optimizer.zero_grad()
            logger.info(f"{bar.postfix}")
        else:
            model.train()
            metric = NERMetric(itos)
            bar = progress_bar(dataloader)
            for i, batch in enumerate(bar, 1):
                words = batch["tokenids"]
                labels = batch["labelids"]
                if self.args.device != '-1':
                    words = words.to(torch.device("cuda"))
                    labels = labels.to(torch.device("cuda"))
                word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
                mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
                score = model(words)
                loss = model.loss(score, labels, mask)
                loss = loss / self.args.update_steps
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if i % self.args.update_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                preds, _ = model.decode(score, mask)
                mask = mask[:, 1:]
                metric(preds.masked_fill(~mask, stoi["O"]), labels)
                bar.set_postfix_str(f"lr: {scheduler.get_last_lr()[0]:.4e} - loss: {loss:.2f} - {metric}")
            optimizer.zero_grad()
            logger.info(f"{bar.postfix}")

    def train(self):
        self.model = CrfSeqTagModel(n_words=AutoTokenizer.from_pretrained(self.args.roberta_path).vocab_size, encoder="bert", n_mlp=800, n_labels=self.labels_num, bert=self.args.roberta_path)
        if self.args.device != '-1':
            self.model = self.model.to(torch.device("cuda"))
        init_logger(logger)
        logger.info(f'{self.model}\n')
        train_set = Dataset(self.args.train_file,
                            self.args.roberta_path,
                            astrain=True)
        train_loader = DataLoader(train_set,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(
                                    RandomSampler(train_set),
                                    batch_size=self.args.batch_size,
                                    drop_last=False,
                                ),
                                num_workers=self.args.n_workers)
        dev_set = Dataset(
            file_name=self.args.dev_file,
            tokenizer_path=self.args.roberta_path,
            astrain=True,
            icsr=self.args.icsr,
        )
        dev_loader = DataLoader(dev_set,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(
                                    SequentialSampler(dev_set),
                                    batch_size=self.args.batch_size,
                                    drop_last=False,
                                ),
                                num_workers=self.args.n_workers)
        test_set = Dataset(
            file_name=self.args.test_file,
            tokenizer_path=self.args.roberta_path,
            astrain=True,
            icsr=self.args.icsr,
        )
        test_loader = DataLoader(test_set,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(
                                    SequentialSampler(test_set),
                                    batch_size=self.args.batch_size,
                                    drop_last=False,
                                ),
                                num_workers=self.args.n_workers)
        logger.info(f"train_set: {len(train_set):6}\n")
        logger.info(f"dev_set: {len(dev_set):6}\n")
        logger.info(f"test_set: {len(test_set):6}\n")
        steps = (len(train_set)//self.args.batch_size) * self.args.epochs // self.args.update_steps
        optimizer = AdamW(
            [{'params': c, 'lr': self.args.lr * (1 if n.startswith('encoder') else self.args.lr_rate)}
             for n, c in self.model.named_parameters()],
            self.args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(steps*self.args.warmup), steps)
        elapsed = timedelta()
        best_e, best_metric = 1, Metric()

        if self.args.use_kl:
            # load the init model
            self.base_model = CrfSeqTagModel(n_words=AutoTokenizer.from_pretrained(self.args.roberta_path).vocab_size, encoder="bert", n_mlp=800, n_labels=self.labels_num, bert=self.args.roberta_path)
            if self.args.device != '-1':
                self.base_model = self.base_model.to(torch.device("cuda"))
            self.base_model.load_state_dict(torch.load(self.args.init_model_path)["state_dict"])
            self.base_model.eval()

        for epoch in range(1, self.args.epochs+1):
            start = datetime.now()
            logger.info(f"Epoch {epoch} / {self.args.epochs}:")
            self._train(train_loader, optimizer, scheduler, train_set.label_itos)
            dev_metric = self._evaluate(dev_loader, dev_set.label_itos)
            logger.info(f" dev: - {dev_metric}")
            test_metric = self._evaluate(test_loader, test_set.label_itos)
            logger.info(f"test: - {test_metric}")
            t = datetime.now() - start
            elapsed += t
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                state = {'state_dict': self.model.state_dict()}
                torch.save(state, os.path.join(self.args.path, 'best.model'))
                logger.info(f"{t}s elapsed, {elapsed}s in total, saved\n")
            else:
                logger.info(f"{t}s elapsed, {elapsed}s in total\n")
            if epoch - best_e >= self.args.patience:
                break
        logger.info(f"Epoch {best_e} saved")
        logger.info(f"{'dev:':5} {best_metric}")

    @torch.no_grad()
    def _evaluate(self, dataloader, itos, model=None):
        stoi = {s: i for i, s in enumerate(itos)}
        if model is None:
            self.model.eval()
            if self.args.onasr:
                metric = LevenNERMetric(itos)
            elif self.args.onlycer:
                metric = LevenMetric()
            else:
                metric = NERMetric(itos)
            bar = progress_bar(dataloader)
            for i, batch in enumerate(bar, 1):
                if self.args.onlycer:
                    gold_s = batch["gold_s"]
                    asr = batch["asr"]
                    metric(asr, gold_s)
                    continue
                words = batch["tokenids"]
                labels = batch["labelids"]
                gold_s = batch["gold_s"]
                asr = batch["asr"]
                ins_labels = batch["ins_labels"]
                if self.args.device != '-1':
                    words = words.to(torch.device("cuda"))
                    labels = labels.to(torch.device("cuda"))
                word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
                mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
                score = self.model(words)
                preds, _ = self.model.decode(score, mask)
                mask = mask[:, 1:]
                if self.args.onasr:
                    metric(preds.masked_fill(~mask, stoi["O"]), labels, asr, gold_s, ins_labels)
                else:
                    metric(preds.masked_fill(~mask, stoi["O"]), labels, ins_labels)
            return metric
        else:
            model.eval()
            if self.args.onasr:
                metric = LevenNERMetric(itos)
            elif self.args.onlycer:
                metric = LevenMetric()
            else:
                metric = NERMetric(itos)
            bar = progress_bar(dataloader)
            for i, batch in enumerate(bar, 1):
                if self.args.onlycer:
                    gold_s = batch["gold_s"]
                    asr = batch["asr"]
                    metric(asr, gold_s)
                    continue
                words = batch["tokenids"]
                labels = batch["labelids"]
                gold_s = batch["gold_s"]
                asr = batch["asr"]
                ins_labels = batch["ins_labels"]
                if self.args.device != '-1':
                    words = words.to(torch.device("cuda"))
                    labels = labels.to(torch.device("cuda"))
                word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
                mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
                score = model(words)
                preds, _ = model.decode(score, mask)
                mask = mask[:, 1:]
                if self.args.onasr:
                    metric(preds.masked_fill(~mask, stoi["O"]), labels, asr, gold_s, ins_labels)
                else:
                    metric(preds.masked_fill(~mask, stoi["O"]), labels, ins_labels)
            return metric

    @torch.no_grad()
    def predict(self):
        self.model = CrfSeqTagModel(n_words=AutoTokenizer.from_pretrained(self.args.roberta_path).vocab_size, encoder="bert", n_mlp=800, n_labels=self.labels_num, bert=self.args.roberta_path)
        if self.args.device != '-1':
            self.model = self.model.to(torch.device("cuda"))
        self.load_model(self.args.path)
        self.model.eval()
        init_logger(logger)
        dataset = Dataset(
            file_name=self.args.input_file,
            tokenizer_path=self.args.roberta_path,
            astrain=False,
        )
        dataloader = DataLoader(dataset,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(
                                    SequentialSampler(dataset),
                                    batch_size=self.args.batch_size,
                                    drop_last=False,
                                ),
                                num_workers=self.args.n_workers)
        logger.info(f"\n dataset: {len(dataset):6}\n")
        logger.info(f"Making predictions on the dataset: {self.args.input_file}")
        bar = progress_bar(dataloader)

        with open(self.args.output_file, 'w', encoding="utf8") as f:
            for batch in bar:
                keys = batch["keys"]
                sents = batch["parse_sents"]
                words = batch["tokenids"]
                if self.args.device != '-1':
                    words = words.to(torch.device("cuda"))

                word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
                mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
                score = self.model(words)
                output, output_p = self.model.decode(score, mask)
                mask = mask[:, 1:]
                output = output.masked_fill(~mask, dataset.label_stoi["O"])
                nes = self.get_nes(output, sents, dataset.label_itos)

                for key, ne, sent, label_seq_p in zip(keys, nes, sents, output_p):
                    this_dic = {"key": key, "text": sent, "label": ne, "label_seq_p": label_seq_p.item()}
                    f.write(json.dumps(this_dic, ensure_ascii=False) + "\n")
                    f.flush()
        logger.info(f"Predictions saved to {self.args.output_file}")

    def evaluate(self):
        self.model = CrfSeqTagModel(n_words=AutoTokenizer.from_pretrained(self.args.roberta_path).vocab_size, encoder="bert", n_mlp=800, n_labels=self.labels_num, bert=self.args.roberta_path)
        if self.args.device != '-1':
            self.model = self.model.to(torch.device("cuda"))
        self.load_model(self.args.path)
        self.model.eval()
        init_logger(logger)
        if self.args.onlycer:
            dataset = Dataset(
            file_name=self.args.input_file,
            tokenizer_path=self.args.roberta_path,
            astrain=False,
            onasr=self.args.onasr,
            icsr=self.args.icsr,)
        else:
            dataset = Dataset(
                file_name=self.args.input_file,
                tokenizer_path=self.args.roberta_path,
                astrain=True,
                onasr=self.args.onasr,
                icsr=self.args.icsr,
            )
        dataloader = DataLoader(dataset,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(
                                    SequentialSampler(dataset),
                                    batch_size=self.args.batch_size,
                                    drop_last=False,
                                ),
                                num_workers=self.args.n_workers)
        
        logger.info(f"\n dataset: {len(dataset):6}\n")
        logger.info(f"Evaluating on the dataset: {self.args.input_file}")
        metric = self._evaluate(dataloader, dataset.label_itos)
        logger.info(f"{'metric:':5} {metric}")

    def get_nes(self, output, sents, itos):
        res = []
        for pred, sent in zip(output, sents):
            label_str_lst = [itos[labelid] for labelid in pred[0:len(sent)].tolist()]
            start = 0
            ne_label = ''
            flag = False
            this_res = []
            for i, label_str in enumerate(label_str_lst):
                if label_str.startswith('B-'):
                    if flag:
                        this_res.append([start, i, ne_label, sent[start:i]])
                    start = i
                    ne_label = label_str[2:]
                    flag = True
                elif label_str.startswith('I'):
                    continue
                else:
                    if flag:
                        this_res.append([start, i, ne_label, sent[start:i]])
                    flag = False
            if flag:
                this_res.append([start, len(sent), ne_label, sent[start:len(sent)]])
            res.append(this_res)

        return res

    def tri_training(self):
        init_logger(logger)
        dev_set = Dataset(
            file_name=self.args.dev_file,
            tokenizer_path=self.args.roberta_path,
            astrain=True,
        )
        dev_loader = DataLoader(dev_set,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(
                                    SequentialSampler(dev_set),
                                    batch_size=self.args.batch_size,
                                    drop_last=False,
                                ),
                                num_workers=self.args.n_workers)
        logger.info(f"\n dev_set: {len(dev_set):6}\n")
        test_set = Dataset(
            file_name=self.args.test_file,
            tokenizer_path=self.args.roberta_path,
            astrain=True,
        )
        test_loader = DataLoader(test_set,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(
                                    SequentialSampler(test_set),
                                    batch_size=self.args.batch_size,
                                    drop_last=False,
                                ),
                                num_workers=self.args.n_workers)
        logger.info(f"\n test_set: {len(test_set):6}\n")
        
        best_iter, best_iter_metric, best_test_iter_metric = 1, Metric(), Metric()
        for iter in range(1, self.args.iter+1):
            # first select the data
            logger.info(f"\ntri-training iteration: {iter} start")
            if os.path.exists(os.path.join(self.args.select_file_dir, f"iter-{iter}.jsonl")):
                logger.info(f"iteration: {iter} data has been selected")
            else:
                self.tri_training_select(iter)
            train_file = os.path.join(self.args.select_file_dir, f"iter-{iter}.jsonl")
            train_set = Dataset(train_file, self.args.roberta_path, astrain=True)
            train_loader = DataLoader(train_set,
                                    collate_fn=collate_fn,
                                    batch_sampler=BatchSampler(
                                        RandomSampler(train_set),
                                        batch_size=self.args.batch_size,
                                        drop_last=False,
                                    ),
                                    num_workers=self.args.n_workers)
            logger.info(f"\n train_set: {len(train_set):6}\n")
            dev_metric, test_metric = self._iter_train(train_loader, dev_loader, test_loader, len(train_set), train_set.label_itos, iter, False)
            logger.info(f"this iteration best dev: - {dev_metric}")
            logger.info(f"this iteration test: - {test_metric}")
            if dev_metric > best_iter_metric:
                best_iter, best_iter_metric, best_test_iter_metric = iter, dev_metric, test_metric
            if iter - best_iter >= self.args.patience:
                break
            else:
                logger.info(f"start training the aff model at iteration {iter}")
                aff_dev_metric, aff_test_metric = self._iter_train(train_loader, dev_loader, test_loader, len(train_set), train_set.label_itos, iter, True)
        logger.info(f"get best dev result at iteration {best_iter}\n")
        logger.info(f"{'dev:':5} {best_iter_metric}")
        logger.info(f"{'test:':5} {best_test_iter_metric}")

    def _iter_train(self, train_loader, dev_loader, test_loader, train_length, itos, iter, if_aff):
        model = CrfSeqTagModel(n_words=AutoTokenizer.from_pretrained(self.args.roberta_path).vocab_size, encoder="bert", n_mlp=800, n_labels=self.labels_num, bert=self.args.roberta_path)
        if self.args.device != '-1':
            model = model.to(torch.device("cuda"))

        steps = (train_length//self.args.batch_size) * self.args.epochs // self.args.update_steps
        optimizer = AdamW(
            [{'params': c, 'lr': self.args.lr * (1 if n.startswith('encoder') else self.args.lr_rate)}
             for n, c in model.named_parameters()],
            self.args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(steps*self.args.warmup), steps)
        elapsed = timedelta()
        best_e, best_metric, best_test_metric = 1, Metric(), Metric()
        for epoch in range(1, self.args.epochs+1):
            start = datetime.now()
            logger.info(f"Epoch {epoch} / {self.args.epochs}:")
            self._train(train_loader, optimizer, scheduler, itos, model)
            dev_metric = self._evaluate(dev_loader, itos, model)
            logger.info(f" dev: - {dev_metric}")
            test_metric = self._evaluate(test_loader, itos, model)
            logger.info(f"test: - {test_metric}")
            t = datetime.now() - start
            elapsed += t
            if dev_metric > best_metric:
                best_e, best_metric, best_test_metric = epoch, dev_metric, test_metric
                state = {'state_dict': model.state_dict()}
                torch.save(state, os.path.join(self.args.path, f"iter-{iter}-{'aff' if if_aff else 'best'}.model"))
                logger.info(f"{t}s elapsed, {elapsed}s in total, saved\n")
            else:
                logger.info(f"{t}s elapsed, {elapsed}s in total\n")
            if epoch - best_e >= self.args.patience:
                break
        logger.info(f"iteration: {iter} end")
        logger.info(f"Epoch {best_e} saved")
        logger.info(f"{'dev:':5} {best_metric}")
        logger.info(f"{'test:':5} {best_test_metric}")
        return best_metric, best_test_metric

    @torch.no_grad()       
    def tri_training_select(self, iter):
        init_logger(logger)

        if iter == 1:
            main_model_path = self.args.init_model_path
            aff_model_path = self.args.init_aff_model_path
        else:
            main_model_path = os.path.join(self.args.path, f"iter-{iter-1}-best.model")
            aff_model_path = os.path.join(self.args.path, f"iter-{iter-1}-aff.model")

        logger.info(f"Selecting the data for tri-training from : {self.args.unlabel_file} with model {main_model_path} and {aff_model_path}")

        main_model = CrfSeqTagModel(n_words=AutoTokenizer.from_pretrained(self.args.roberta_path).vocab_size, encoder="bert", n_mlp=800, n_labels=self.labels_num, bert=self.args.roberta_path)
        aff_model = CrfSeqTagModel(n_words=AutoTokenizer.from_pretrained(self.args.roberta_path).vocab_size, encoder="bert", n_mlp=800, n_labels=self.labels_num, bert=self.args.roberta_path)
        if self.args.device != '-1':
            main_model = main_model.to(torch.device("cuda"))
            aff_model = aff_model.to(torch.device("cuda"))
        main_model.load_state_dict(torch.load(main_model_path)["state_dict"])
        aff_model.load_state_dict(torch.load(aff_model_path)["state_dict"])
        main_model.eval()
        aff_model.eval()

        dataset = Dataset(
            file_name=self.args.unlabel_file,
            tokenizer_path=self.args.roberta_path,
            astrain=False,
        )
        dataloader = DataLoader(dataset,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(
                                    SequentialSampler(dataset),
                                    batch_size=self.args.batch_size,
                                    drop_last=False,
                                ),
                                num_workers=self.args.n_workers)
        logger.info(f"\n all unlabel dataset: {len(dataset):6}\n")
        bar = progress_bar(dataloader)
        sum_selected = 0
        if not os.path.exists(self.args.select_file_dir):
            os.mkdir(self.args.select_file_dir)

        with open(os.path.join(self.args.select_file_dir, f"iter-{iter}.jsonl"), 'w', encoding="utf8") as f:
            # write the src data
            src_data = read_json(self.args.src_train_file)
            for dic in src_data:
                f.write(json.dumps(dic, ensure_ascii=False) + "\n")

            for batch in bar:
                keys = batch["keys"]
                sents = batch["sents"]
                words = batch["tokenids"]
                if self.args.device != '-1':
                    words = words.to(torch.device("cuda"))

                word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
                mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
                main_score = main_model(words)
                aff_score = aff_model(words)
                main_output, main_output_p = main_model.decode(main_score, mask)
                aff_output, aff_output_p = aff_model.decode(aff_score, mask)
                mask = mask[:, 1:]
                main_output = main_output.masked_fill(~mask, dataset.label_stoi["O"])
                aff_output = aff_output.masked_fill(~mask, dataset.label_stoi["O"])

                # select the data that has the same label
                # [batch_size]
                same_mask = (main_output == aff_output).all(-1)
                sum_selected += same_mask.sum().item()

                nes = self.get_nes(main_output, sents, dataset.label_itos)

                for key, ne, sent, select in zip(keys, nes, sents, same_mask):
                    if select:
                        this_dic = {"key": key, "text": sent, "label": ne}
                        f.write(json.dumps(this_dic, ensure_ascii=False) + "\n")
                        f.flush()
        
        logger.info(f"Selected {sum_selected} examples, ratio: {sum_selected / len(dataset):.4f}, combined with src train {self.args.src_train_file}, saved to {os.path.join(self.args.select_file_dir, f'iter-{iter}.jsonl')}.")

    @torch.no_grad()
    def stt_select_data(self, iter: int, best_iter = None):
        # load last best model
        s_model = CrfSeqTagModel(n_words=AutoTokenizer.from_pretrained(self.args.roberta_path).vocab_size, encoder="bert", n_mlp=800, n_labels=self.labels_num, bert=self.args.roberta_path)
        if self.args.device != '-1':
            s_model = s_model.to(torch.device("cuda"))
        if iter > 1:
            model_path = os.path.join(self.args.path, f"iter-{best_iter}-best.model")
        else:
            model_path = self.args.init_model_path
        s_model.load_state_dict(torch.load(model_path)["state_dict"])
        s_model.eval()

        init_logger(logger)
        selected_keys = set()
        selected_tansformed_data = []
        transformed_data = []
        if iter <= self.args.trans_iter:
            logger.info(f"Using data from the predicted transformed data by baseline")
            # is predicted by the baseline model
            transformed_data = read_json(self.args.prd_transformed_data)
            # avoid the same data
            # remove the same data and for that has no empty label sample at rate
            for dic in transformed_data:
                key = dic["key"]
                if key in selected_keys:
                    continue
                if dic["label"] != [] and dic["label_seq_p"] > self.args.p_hold:
                    selected_keys.add(key)
                    selected_tansformed_data.append(dic)
                else:
                    if "label_seq_p" in dic:
                        if torch.rand(1) > self.args.p_drop and dic["label_seq_p"] > self.args.p_hold:
                            selected_keys.add(key)
                            selected_tansformed_data.append(dic)
            logger.info(f"{len(selected_keys)} examples selected from the predicted transformed data {self.args.prd_transformed_data}")

        dataset = Dataset(
            file_name=self.args.unlabel_file,
            tokenizer_path=self.args.roberta_path,
            astrain=False,
            filter_keys=selected_keys,
        )
        dataloader = DataLoader(dataset,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(
                                    SequentialSampler(dataset),
                                    batch_size=self.args.batch_size,
                                    drop_last=False,
                                ),
                                num_workers=self.args.n_workers)
        logger.info(f"\n all unlabel dataset: {len(dataset):6} (except the {len(selected_keys)} ones selected in predicted transfomed data)\n")
        logger.info(f"Selecting the data for self-transform-training from : {self.args.unlabel_file} with model {model_path}")

        bar = progress_bar(dataloader)
        sum_selected = 0
        if not os.path.exists(self.args.select_file_dir):
            os.mkdir(self.args.select_file_dir)

        with open(os.path.join(self.args.select_file_dir, f"iter-{iter}.jsonl"), 'w', encoding="utf8") as f:
            # write the src data
            src_data = read_json(self.args.src_train_file)
            for dic in src_data:
                f.write(json.dumps(dic, ensure_ascii=False) + "\n")
            # write the predicted transformed data if any
            for dic in selected_tansformed_data:
                f.write(json.dumps(dic, ensure_ascii=False) + "\n")
            for batch in bar:
                keys = batch["keys"]
                sents = batch["sents"]
                words = batch["tokenids"]
                if self.args.device != '-1':
                    words = words.to(torch.device("cuda"))

                word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
                mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
                score = s_model(words)
                output, output_p = s_model.decode(score, mask)
                mask = mask[:, 1:]
                output = output.masked_fill(~mask, dataset.label_stoi["O"])
                p_large_p_hold = output_p.ge(self.args.p_hold)
                all_o = output.eq(dataset.label_stoi["O"]).all(-1)
                random_p = torch.rand_like(all_o, dtype=torch.float)
                allo_select_mask = p_large_p_hold & all_o & (random_p > self.args.p_drop)
                other_select_mask = p_large_p_hold & (~all_o)
                select_mask = allo_select_mask | other_select_mask
                sum_selected += select_mask.sum().item()

                nes = self.get_nes(output, sents, dataset.label_itos)
                for key, ne, sent, select in zip(keys, nes, sents, select_mask):
                    if select:
                        this_dic = {"key": key, "text": sent, "label": ne}
                        f.write(json.dumps(this_dic, ensure_ascii=False) + "\n")
                        f.flush()
        
        logger.info(f"Select {sum_selected} examples by the model {model_path}, {len(selected_keys)} examples from the predicted transformed file {self.args.prd_transformed_data}, sum: {sum_selected+len(selected_keys)}, combined with src train {self.args.src_train_file}, saved to {os.path.join(self.args.select_file_dir, f'iter-{iter}.jsonl')}.")
        with open(os.path.join(self.args.select_file_dir, f"iter-{iter}.finish"), 'w', encoding="utf8") as f:
            f.write(f"{iter} data selecting finished. \n")
            f.write(f"Select {sum_selected} examples by the model {model_path}, {len(selected_keys)} examples from the predicted transformed file {self.args.prd_transformed_data}, sum: {sum_selected+len(selected_keys)}, combined with src train {self.args.src_train_file}, saved to {os.path.join(self.args.select_file_dir, f'iter-{iter}.jsonl')}.")

    def self_trans_training(self):
        init_logger(logger)
        dev_set = Dataset(
            file_name=self.args.dev_file,
            tokenizer_path=self.args.roberta_path,
            astrain=True,
        )
        dev_loader = DataLoader(dev_set,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(
                                    SequentialSampler(dev_set),
                                    batch_size=self.args.batch_size,
                                    drop_last=False,
                                ),
                                num_workers=self.args.n_workers)
        logger.info(f"\n dev_set: {len(dev_set):6}\n")
        test_set = Dataset(
            file_name=self.args.test_file,
            tokenizer_path=self.args.roberta_path,
            astrain=True,
        )
        test_loader = DataLoader(test_set,
                                collate_fn=collate_fn,
                                batch_sampler=BatchSampler(
                                    SequentialSampler(test_set),
                                    batch_size=self.args.batch_size,
                                    drop_last=False,
                                ),
                                num_workers=self.args.n_workers)
        logger.info(f"\n test_set: {len(test_set):6}\n")

        best_iter, best_iter_metric = 1, Metric()

        if self.args.use_kl:
            # load the init model
            self.base_model = CrfSeqTagModel(n_words=AutoTokenizer.from_pretrained(self.args.roberta_path).vocab_size, encoder="bert", n_mlp=800, n_labels=self.labels_num, bert=self.args.roberta_path)
            if self.args.device != '-1':
                self.base_model = self.base_model.to(torch.device("cuda"))
            self.base_model.load_state_dict(torch.load(self.args.init_model_path)["state_dict"])
            self.base_model.eval()

        for iter in range(1, self.args.iter+1):
            # first use the last best model to select the unlabeled data
            logger.info(f"\nself transform training iteration: {iter} start")
            if os.path.exists(os.path.join(self.args.select_file_dir, f"iter-{iter}.finish")):
                logger.info(f"iteration: {iter} data has been selected")
            else:
                self.stt_select_data(iter, best_iter)
            train_file = os.path.join(self.args.select_file_dir, f"iter-{iter}.jsonl")
            train_set = Dataset(train_file, self.args.roberta_path, astrain=True)
            train_loader = DataLoader(train_set,
                                    collate_fn=collate_fn,
                                    batch_sampler=BatchSampler(
                                        RandomSampler(train_set),
                                        batch_size=self.args.batch_size,
                                        drop_last=False,
                                    ),
                                    num_workers=self.args.n_workers)
            logger.info(f"\n train_set: {len(train_set):6}\n")

            # from stratch
            self.model = CrfSeqTagModel(n_words=AutoTokenizer.from_pretrained(self.args.roberta_path).vocab_size, encoder="bert", n_mlp=800, n_labels=self.labels_num, bert=self.args.roberta_path)
            if self.args.device != '-1':
                self.model = self.model.to(torch.device("cuda"))

            steps = (len(train_set)//self.args.batch_size) * self.args.epochs // self.args.update_steps
            optimizer = AdamW(
                [{'params': c, 'lr': self.args.lr * (1 if n.startswith('encoder') else self.args.lr_rate)}
                 for n, c in self.model.named_parameters()],
                self.args.lr)
            scheduler = get_linear_schedule_with_warmup(optimizer, int(steps*self.args.warmup), steps)

            elapsed = timedelta()
            best_e, best_metric = 1, Metric()
            for epoch in range(1, self.args.epochs+1):
                start = datetime.now()
                logger.info(f"Epoch {epoch} / {self.args.epochs}:")
                self._train(train_loader, optimizer, scheduler, train_set.label_itos)
                dev_metric = self._evaluate(dev_loader, dev_set.label_itos)
                logger.info(f" dev: - {dev_metric}")
                test_metric = self._evaluate(test_loader, test_set.label_itos)
                logger.info(f"test: - {test_metric}")
                t = datetime.now() - start
                elapsed += t
                if dev_metric > best_metric:
                    best_e, best_metric = epoch, dev_metric
                    state = {'state_dict': self.model.state_dict()}
                    torch.save(state, os.path.join(self.args.path, f"iter-{iter}-best.model"))
                    logger.info(f"{t}s elapsed, {elapsed}s in total, saved\n")
                else:
                    logger.info(f"{t}s elapsed, {elapsed}s in total\n")
                if epoch - best_e >= self.args.patience:
                    break
            logger.info(f"iteration: {iter} end")
            logger.info(f"Epoch {best_e} saved")
            logger.info(f"{'dev:':5} {best_metric}")

            if best_metric > best_iter_metric:
                best_iter, best_iter_metric = iter, best_metric
            if iter - best_iter >= self.args.patience:
                break
        logger.info(f"get best dev result at iteration {best_iter}\n")
        logger.info(f"{'dev:':5} {best_iter_metric}")

class InsParser(object):
    def __init__(self, args):
        self.args = args
        self.labels_num = 5
    
    def load_model(self, model_path):
        state = torch.load(model_path)
        # args = state['args']
        # if args.bert is not None:
        #     args.bert = self.args.roberta_path
        # args.bert_req_grad = self.args.bert_req_grad
        # model = CrfSeqTagModel(**args)
        # if state['pretrained'] is not None:
        #     model.load_pretrained(state['pretrained'])
        self.model.load_state_dict(state['state_dict'], False)

    def train(self):
        self.model = SentLabelModel(n_words=AutoTokenizer.from_pretrained(self.args.roberta_path).vocab_size, encoder="bert", n_mlp=800, n_labels=self.labels_num, bert=self.args.roberta_path, bert_req_grad=True)
        if self.args.device != '-1':
            self.model = self.model.to(torch.device("cuda"))
        init_logger(logger)
        logger.info(f'{self.model}\n')
        train_set = InsDataset(self.args.train_file,
                            self.args.roberta_path,
                            astrain=True)
        train_loader = DataLoader(train_set,
                                collate_fn=ins_collate_fn,
                                batch_sampler=BatchSampler(
                                    RandomSampler(train_set),
                                    batch_size=self.args.batch_size,
                                    drop_last=False,
                                ),
                                num_workers=self.args.n_workers)
        dev_set = InsDataset(
            file_name=self.args.dev_file,
            tokenizer_path=self.args.roberta_path,
            astrain=True,
        )
        dev_loader = DataLoader(dev_set,
                                collate_fn=ins_collate_fn,
                                batch_sampler=BatchSampler(
                                    SequentialSampler(dev_set),
                                    batch_size=self.args.batch_size,
                                    drop_last=False,
                                ),
                                num_workers=self.args.n_workers)
        test_set = InsDataset(
            file_name=self.args.test_file,
            tokenizer_path=self.args.roberta_path,
            astrain=True,
        )
        test_loader = DataLoader(test_set,
                                collate_fn=ins_collate_fn,
                                batch_sampler=BatchSampler(
                                    SequentialSampler(test_set),
                                    batch_size=self.args.batch_size,
                                    drop_last=False,
                                ),
                                num_workers=self.args.n_workers)
        logger.info(f"train_set: {len(train_set):6}\n")
        logger.info(f"dev_set: {len(dev_set):6}\n")
        logger.info(f"test_set: {len(test_set):6}\n")
        steps = (len(train_set)//self.args.batch_size) * self.args.epochs // self.args.update_steps
        optimizer = AdamW(
            [{'params': c, 'lr': self.args.lr * (1 if n.startswith('encoder') else self.args.lr_rate)}
             for n, c in self.model.named_parameters()],
            self.args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(steps*self.args.warmup), steps)
        elapsed = timedelta()
        best_e, best_metric = 1, Metric()

        for epoch in range(1, self.args.epochs+1):
            start = datetime.now()
            logger.info(f"Epoch {epoch} / {self.args.epochs}:")
            self._train(train_loader, optimizer, scheduler, train_set.label_itos)

    def _train(self, dataloader, optimizer, scheduler):
        pass