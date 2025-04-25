import torch
from model import *
from config import *
from attention import *


for batch in train_data_loader:
    first_batch = batch
    break  # Just to check the first batch


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


input_tensor = first_batch['trg_ids']
encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
method = None
align_method = "dot"
timestep = None

predictions, hidden, attn_weights = decoder(
    input=input_tensor, 
    encoder_outputs=encoder_outputs, 
    hidden=hidden, 
    align_method=align_method,
    method=method,
    timestep=timestep
    )

print(predictions.shape, hidden.shape, attn_weights.shape)


