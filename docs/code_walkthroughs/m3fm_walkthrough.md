# M3FM Code Walkthrough

> **KB references:** Model card (pending) · [Integration strategy](../integration/integration_strategy.md) · [Experiment config stub](../kb/templates/experiment_config_stub.md)

## Overview
M3FM couples multilingual CLIP text embeddings with the original R2Gen relational-memory Transformer decoder to generate bilingual COVID-era chest X-ray reports. The entrypoint `M3FM.py` wires tokenization, dataset splits, optimizer/scheduler, and the trainer while `modules/text_extractor.py` handles medical text preprocessing and embedding, and `modules/encoder_decoder.py` implements the Transformer + RelationalMemory decoder that outputs report logits for teacher-forced training; inference routes beam/greedy decoding through English or Chinese heads via CLI flags.^[```1:72:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/README.md```][```15:126:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/M3FM.py```][```16:53:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/modules/text_extractor.py```][```227:355:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/modules/encoder_decoder.py```][```130:210:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/inference.py```]

## At-a-Glance
| Architecture | Params | Context | Inputs | Key capabilities | Repo |
| --- | --- | --- | --- | --- | --- |
| Multilingual CLIP text embeddings → relational-memory Transformer decoder (beam/greedy) for bilingual CXRs.^[```16:53:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/modules/text_extractor.py```][```227:355:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/modules/encoder_decoder.py```][```130:210:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/inference.py```] | Defaults: `d_model=512`, FFN 2048, 3 decoder layers, 8 heads, `rm_num_slots=3`, `beam_size=3`, `epochs=15`.^[```29:81:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/M3FM.py```][```34:76:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/inference.py```] | 224×224 CXRs with max 100 tokens; BOS token `1` (English) or `2` (Chinese) selects the generation language.^[```18:75:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/modules/dataloaders.py```][```20:115:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/modules/datasets.py```][```162:210:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/inference.py```] | `annotation.json` + image roots streamed by `R2DataLoader`, yielding `(reports_ids, reports_ids_use)` tensors for teacher forcing.^[```18:75:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/modules/dataloaders.py```][```20:115:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/modules/datasets.py```] | Trainer wraps SGD + StepLR, gradient clipping, multilingual greedy decoding, and BLEU/SPICE-compatible evaluation utilities.^[```91:124:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/M3FM.py```][```203:221:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/modules/trainer.py```][```130:210:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/inference.py```] | [github.com/ai-in-health/M3FM](https://github.com/ai-in-health/M3FM) |

### Environment & Hardware Notes
- **Conda + pip workflow.** Create `conda create -n M3FM python==3.9`, activate, install CUDA 11.8-compatible PyTorch (`torch>=1.10.1`, `torchvision>=0.11.2`, `pytorch-cuda==11.8`) followed by `pip install -r requirements.txt`; repo validated on `torch==2.2.1`.^[```4:21:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/README.md```]
- **Metric prerequisites.** Java, `pycocoevalcap`, `pycocotools`, and Stanford CoreNLP jars are required for SPICE; README documents manual download/placement steps to avoid firewalls.^[```46:71:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/README.md```]
- **Language evaluation assets.** Place `stanford-corenlp-4.5.2` under `data/` and keep `corenlp_root` in `configs/__init__.py` synchronized when switching between English and Chinese inference.^[```61:71:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/README.md```]

## Key Components

### Tokenizer + Data Interface (`modules/dataloaders.py`, `modules/datasets.py`)
`R2DataLoader` centralizes resizing/normalization, dataset selection (IU X-Ray vs. MIMIC/COV), and a collate function that pads both the teacher-forced `reports_ids` (targets) and decoder inputs (`reports_ids_use`). The dataset class uses cleaned strings to build token IDs, tracks language label via the leading token, and emits both full targets and shifted inputs.

```8:45:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/modules/dataloaders.py
class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
```

```48:74:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/modules/dataloaders.py
    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, report, seq_lengths, seq_length1,image_path_all, reports_ids_use = zip(*data)

        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        max_seq_length_us = max(seq_length1)
        targets_us = np.zeros((len(reports_ids_use),  max_seq_length_us), dtype=int)
        targets_masks1 = np.zeros((len(reports_ids_use), max_seq_length_us ), dtype=int)

        for i,reports_ids_use1 in enumerate(reports_ids_use):
            targets_us[i, :len(reports_ids_use1)] = reports_ids_use1

        return images_id, images, torch.LongTensor(targets), report,image_path_all ,torch.LongTensor(targets_us)
```

### Multilingual TextExtractor (`modules/text_extractor.py`)
The `TextExtractor` loads `M-CLIP/XLM-Roberta-Large-Vit-L-14`, averages contextual token embeddings with attention masking, projects them through CLIP’s linear head, then applies a learnable affine + ReLU to map the 768-d output to the 512-d hidden size expected by the decoder. Reports are cleaned per language before tokenization, enabling bilingual support without retraining the encoder.

```16:53:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/modules/text_extractor.py
class TextExtractor(nn.Module):
    def __init__(self, args):
        super(TextExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
        self.model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(self.model_name, device=self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, device=self.device)
        self.clean_report = self.clean_report_cov
        
        self.transformer = self.model.transformer
        self.LinearTransformation = self.model.LinearTransformation
        
        self.affine_aa = nn.Linear(768, 512).cuda()
        
    def forward(self, reports):
        if isinstance(reports, tuple):
            texts=[]
            for example in reports:
                example=self.clean_report(example)
                texts.append(example)
        else:
            texts=self.clean_report(reports)


        with torch.no_grad():
            txt_tok = self.tokenizer(texts, padding=True, return_tensors='pt').to(self.device)
            embs = self.transformer(**txt_tok)[0]
            att = txt_tok['attention_mask']
            embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
            embeddings = self.LinearTransformation(embs).cuda()

        


        embeddings = F.relu(self.affine_aa(embeddings)).cuda() #batch*768--》batch*512
        return embeddings #batch*512
```

### Relational-Memory Transformer Decoder (`modules/encoder_decoder.py`)
`Transformer` wraps a Decoder-only stack augmented with conditional layer norm controlled by a relational memory module. Before projection, the model reshapes logits to match token vocab (default 464). Memory slots capture long-range dependencies from the previous tokens, improving report fluency over vanilla Transformer decoders.

```228:252:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/modules/encoder_decoder.py
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        #self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()
        self.rm=RelationalMemory()
        self.tgt_emb = Embeddings().cuda()

    def forward(self, enc_outputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        dec_outputs = self.decode(dec_inputs,  enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        
        return dec_logits.reshape(-1, dec_logits.size(-1))

    def decode(self,  dec_inputs, enc_outputs):
        memory = self.rm.init_memory(enc_outputs.size(0)).to( enc_outputs)
        memory = self.rm(self.tgt_emb(dec_inputs), memory)
        return self.decoder(dec_inputs, enc_outputs, memory)
```

### Trainer & Scheduler (`modules/trainer.py`)
`Trainer._train_epoch` streams teacher-forced batches, clips gradients to `0.1`, steps SGD and the StepLR schedule every iteration, and records average loss per epoch. Mixed precision isn’t enabled here, so plan GPU memory accordingly.

```203:221:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/modules/trainer.py
    def _train_epoch(self, epoch):

        train_loss = 0
        self.model.cuda().train()
        for batch_idx, (images_id, images, reports_ids, report,image_path_all,reports_ids_use) in enumerate(self.train_dataloader):
            images, reports_ids,reports_ids_use= images.to(self.device), reports_ids.to(self.device),reports_ids_use.to(self.device)

            output = self.model(report, reports_ids_use)
            
            loss = self.criterion(output, reports_ids[:, 1:].reshape(-1))
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.lr_scheduler.step()
        return log
```

### Bilingual Inference Script (`inference.py`)
`inference.py` mirrors the training CLI, loads both English and Chinese `R2GenModel` variants, performs greedy decoding conditioned on the BOS token, and prints generated reports. Changing `--language` toggles which head runs and when the search halts.

```162:210:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/inference.py
if args.language=='English' or args.language=='All':
    model_en.eval()
    output = False
    with torch.no_grad():
        for batch_idx, (images_id, images, reports_ids, report, image_path_all, reports_ids_use) in enumerate(
                test_dataloader):
            images, reports_ids, reports_ids_use = images.to(device), reports_ids.to(
                device), reports_ids_use.to(device)

            for i in range(len(images_id)):
                if reports_ids[i][0] == 1:
                    greedy_dec_input = greedy_decoder(model_en, image_path_all[i], reports_ids[i], start_symbol=1)
                    predict = model_en(image_path_all[i], greedy_dec_input)
                    predict = predict.data.max(1, keepdim=True)[1]
                    output = True
                    predict = predict.squeeze()
                    report = model_en.tokenizer.decode(predict.cpu().numpy())
                    print("----------------------------------------------------------------------------------------")
                    print("Generated English Report:")
                    print(report)
                    print("----------------------------------------------------------------------------------------")
                    break
            if output:
                break
```

## Integration Hooks (Vision ↔ Clinical Language)
- **Tap 512-d text embeddings.** `TextExtractor` already outputs normalized 512-d vectors before relational memory; export them for multimodal alignment (e.g., with genetic embeddings) or to seed cross-modal contrastive losses.^[```16:53:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/modules/text_extractor.py```]
- **Language-aware batching.** The dataset prefixes each sequence with `1` (English) or `2` (Chinese); filtering on the first token lets you run per-language evaluators or remap tokens to KB-friendly ontologies without retraining.^[```20:75:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/modules/datasets.py```]
- **Reuse greedy decoder outputs.** `greedy_decoder` yields raw token IDs before detokenization, making it straightforward to log intermediate logits or route them into KB experiment trackers for BLEU/SPICE comparisons across modalities.^[```130:210:/Users/allison/Projects/neuro-omics-kb/external_repos/M3FM/inference.py```]

