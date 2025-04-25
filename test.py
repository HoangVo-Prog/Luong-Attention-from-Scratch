import torch
from model import *
from config import *
from attention import *


for batch in train_data_loader:
    first_batch = batch
    break  # Just to check the first batch

encoder = Encoder(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    embedding_dim=EMBEDDING_DIM, 
    output_dim=OUTPUT_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL_ENCODER
)


decoder = Decoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL_DECODER,
        attention_type='global'  # or 'local'
    )


seq2seq = Seq2Seq(encoder, decoder)

outputs = seq2seq(
    src=batch["src_ids"], 
    trg=batch["trg_ids"], 
    teacher_forcing_ratio=TEACHER_FORCING_RATIO, 
    align_method="dot",
    method=None    
)


print(outputs.shape)