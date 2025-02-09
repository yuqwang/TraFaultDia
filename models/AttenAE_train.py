import torch
from torch.utils.data import DataLoader
import numpy as np
from functools import partial
from torch import nn
import argparse
import os
from transformers import BertModel, BertTokenizer, BertConfig


from DatasetFusion import DatasetFusion
from AttenAE import AttenAEFusionModel
from torch.cuda.amp import autocast, GradScaler
import random
from torch.optim import AdamW


# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    from transformers import set_seed
    set_seed(seed)

# mp.set_start_method('spawn', force=True)
def create_bert_sentence_emb(sentences, type):
    if type == 'spanurl':
        max_length = 17
    if type == 'log':
        max_length = 16
        # Set cache batch size depending on GPU memory
    cache_size = 400
    embeddings = []

    model.eval()

    for i in range(0, len(sentences), cache_size):
        print(f"Processing bert setence batch starting at index {i}", flush=True)
        batch_sentences = sentences[i: i + cache_size]
        # Tokenize sentences in current batch
        tokenized_batches = tokenizer(batch_sentences, truncation=True, padding='max_length', add_special_tokens=True,
                                      return_tensors='pt', max_length=max_length)

        # Move tokenized input of current batch to GPU
        tokens_tensor = tokenized_batches['input_ids'].to(device, non_blocking=True)
        attention_mask = tokenized_batches['attention_mask'].to(device, non_blocking=True)

        with torch.no_grad():
            print(f"Getting outputs for batch starting at index {i}", flush=True)
            outputs = model(tokens_tensor, attention_mask=attention_mask)

        hidden_states = outputs[2]
        token_vecs = hidden_states[-1]
        sentence_embs = torch.mean(token_vecs, dim=1)

        embeddings.append(sentence_embs)
        torch.cuda.empty_cache()
    embeddings = torch.cat(embeddings, dim=0)

    # del tokenized_batches
    # del outputs
    # del token_vecs

    return embeddings


def collate_fn(batch, spans_all_df, spans_emb, logs_all_df, logs_emb, span_max_length, log_max_length):
    spans_batch_tensor = torch.zeros(len(batch), log_max_length, 775, device=device)
    logs_batch_tensor = torch.zeros(len(batch), log_max_length, 768, device=device)

    for index, sample in enumerate(batch, start=0):
        spans, logs = sample
        span_indexes = spans_all_df[spans_all_df['TraceId'] == spans[:, 2][0]].index
        trace_spans_emb_tensor = spans_emb[span_indexes]

        spans_features_np = spans[:, 4:].astype(np.float32)
        spans_features_tensor = torch.tensor(spans_features_np, device=device)

        spans_tensor = torch.cat((spans_features_tensor, trace_spans_emb_tensor), dim=1)
        spans_batch_tensor[index, :spans.shape[0], :] = spans_tensor

        log_indexes = logs_all_df[logs_all_df['TraceId'] == spans[:, 2][0]].index
        trace_logMsgs_tensor = logs_emb[log_indexes]
        logs_batch_tensor[index, :logs.shape[0], :] = trace_logMsgs_tensor
    return spans_batch_tensor, logs_batch_tensor


if __name__ == '__main__':
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("train_model_54881_00003")
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    #trainticket example
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--embed_dim', type=int, help='embed_dim', default=768)
    argparser.add_argument('--span_input_dim', type=int, help='span_input_dim', default=771)
    argparser.add_argument('--log_input_dim', type=int, help='log_input_dim', default=768)
    argparser.add_argument('--span_max_length', type=int, help='span_max_length', default=91)
    argparser.add_argument('--log_max_length', type=int, help='log_max_length', default=194)
    argparser.add_argument('--num_heads', type=int, help='num_heads', default=8)
    argparser.add_argument('--dropout_rate', type=int, help='dropout_rate', default=0.14857704629135002)
    argparser.add_argument('--batch_size', type=int, help='batch_size', default=32)
    argparser.add_argument('--train_ntrace', type=int, help='train_ntrace', default=3360)
    # 8800, 1536
    argparser.add_argument('--test_ntrace', type=int, help='test_ntrace', default=570)
    argparser.add_argument('--lr', type=int, help='lr', default=3.5496680639743535e-05)
    argparser.add_argument('--epochs', type=int, help='epochs', default=23)

    args = argparser.parse_args()

    train_dataset = DatasetFusion(datamodel="train", num_traces=args.train_ntrace, span_max_length=args.span_max_length,
                                  log_max_length=args.log_max_length)
    print("Train dataset created", flush=True)

    train_spans_df, train_logs_df = train_dataset.get_full_data()
    print("Full data fetched for training", flush=True)

    train_spanurls = train_spans_df['URL'].tolist()
    train_spans_emb = create_bert_sentence_emb(train_spanurls, 'spanurl')
    print("step 1", flush=True)

    train_logMsgs = train_logs_df['LogMsgFull'].tolist()
    train_logs_emb = create_bert_sentence_emb(train_logMsgs, 'log')
    print("step 2", flush=True)

    train_cus_collate = partial(collate_fn, spans_all_df=train_spans_df, spans_emb=train_spans_emb,
                                logs_all_df=train_logs_df, logs_emb=train_logs_emb, span_max_length=args.span_max_length,
                                log_max_length=args.log_max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_cus_collate,
                                  shuffle=True)

    print("About to get validation dataset", flush=True)
    val_dataset = DatasetFusion(datamodel="val", num_traces=args.test_ntrace, span_max_length=args.span_max_length,
                                log_max_length=args.log_max_length)
    print("Validation dataset created", flush=True)

    val_spans_df, val_logs_df = val_dataset.get_full_data()
    val_spanurls = val_spans_df['URL'].tolist()
    val_spans_emb = create_bert_sentence_emb(val_spanurls, 'spanurl')
    print("step 3", flush=True)
    val_logMsgs = val_logs_df['LogMsgFull'].tolist()
    val_logs_emb = create_bert_sentence_emb(val_logMsgs, 'log')

    print("step 4", flush=True)
    val_cus_collate = partial(collate_fn, spans_all_df=val_spans_df, spans_emb=val_spans_emb,
                              logs_all_df=val_logs_df, logs_emb=val_logs_emb, span_max_length=args.span_max_length,
                              log_max_length=args.log_max_length)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=val_cus_collate,
                                shuffle=False)

    config = {
        "dropout_rate": args.dropout_rate,
        "embed_dim": args.embed_dim,
        "span_input_dim": args.span_input_dim,
        "log_input_dim": args.log_input_dim,
        "num_heads": args.num_heads,
        "span_max_length": args.span_max_length,
        "log_max_length":args.log_max_length
    }

    MultiHeadFusion_model = AttenAEFusionModel(config).to(device)

    if torch.cuda.device_count() > 1:
        MultiHeadFusion_model = nn.DataParallel(MultiHeadFusion_model)

    criterion = nn.MSELoss()

    #optimizer = torch.optim.Adam(MultiHeadFusion_model.parameters(), lr=args.lr)
    optimizer = AdamW(MultiHeadFusion_model.parameters(), lr=args.lr)

    epochs = args.epochs

    scaler = GradScaler()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"Starting epoch {epoch}")
        running_loss = 0.0

        MultiHeadFusion_model.train()

        for i, (train_current_spans_batch, train_current_logs_batch) in enumerate(train_dataloader):
            # Forward and backward propagation
            optimizer.zero_grad()
            with autocast():
                recon_spans, recon_logs, fusion_output = MultiHeadFusion_model(train_current_spans_batch,
                                                                               train_current_logs_batch)
                loss1 = criterion(recon_spans, train_current_spans_batch)
                loss2 = criterion(recon_logs, train_current_logs_batch)
                loss = loss1 + loss2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            torch.cuda.empty_cache()

        validation_loss = 0.0

        MultiHeadFusion_model.eval()
        for val_current_spans_batch, val_current_logs_batch in val_dataloader:
            with autocast():
                recon_spans, recon_logs, fusion_output = MultiHeadFusion_model(val_current_spans_batch,
                                                                               val_current_logs_batch)
                loss1 = criterion(recon_spans, val_current_spans_batch)
                loss2 = criterion(recon_logs, val_current_logs_batch)
                loss = loss1 + loss2

            validation_loss += loss.item()
            torch.cuda.empty_cache()

        avg_val_loss = validation_loss / len(val_dataloader)
        print('Epoch [{}/{}], Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(
            epoch + 1, epochs, running_loss / len(train_dataloader), avg_val_loss), flush=True)

        # Save the model if it has the best validation loss so far
        if not os.path.exists('./TTfusionModels'):
            os.makedirs('./TTfusionModels')

        save_path = f'./TTfusionModels/_best_model.pth'
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best model saved at epoch {epoch + 1}!")  # Add this line to print the epoch
            torch.save(MultiHeadFusion_model.state_dict(), save_path)



