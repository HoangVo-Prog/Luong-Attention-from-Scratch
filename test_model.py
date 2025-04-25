import torch
import test_model
from model import *
from config import *
from attention import *
import pytest
import itertools

for batch in train_data_loader:
    first_batch = batch
    break  # Just to check the first batch


def test_encoder_output_shapes():
    global first_batch
    
    encoder = Encoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL_ENCODER
    )
    input_tensor = first_batch['src_ids']
    output, hidden = encoder(input_tensor)
    assert output.shape == (BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)
    assert hidden.shape == (BATCH_SIZE, HIDDEN_DIM)


@pytest.mark.parametrize("align_method", ["dot", "general", "concat"])
def test_global_attention_shapes(align_method):
    global first_batch
    
    attention = GlobalAttention(hidden_dim=HIDDEN_DIM)
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
    context, attn_weights = attention(encoder_outputs, hidden, align_method)
    assert context.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert attn_weights.shape == (BATCH_SIZE, MAX_LENGTH)
    

@pytest.mark.parametrize(
    "method, align_method",
    itertools.product(["monotonic", "predictive"], ["dot", "general", "concat"])
)
def test_local_attention_shapes(method, align_method):
    global first_batch
    
    attention = LocalAttention(hidden_dim=HIDDEN_DIM)
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
    timestep = torch.randint(0, MAX_LENGTH, (BATCH_SIZE,))  # Random timestep for testing
    context, attn_weights = attention(encoder_outputs, hidden, align_method, method, timestep)
    assert context.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert attn_weights.shape == (BATCH_SIZE, MAX_LENGTH)
       
    
def test_global_attention_exception_shapes():
    global first_batch
    
    attention = GlobalAttention(hidden_dim=HIDDEN_DIM)
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
    with pytest.raises(ValueError):
        _, _ = attention(encoder_outputs, hidden, align_method="invalid_method")
        
        
def test_local_attention_exception_align_method_shapes():
    global first_batch
    
    attention = LocalAttention(hidden_dim=HIDDEN_DIM)
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
    timestep = torch.randint(0, MAX_LENGTH, (BATCH_SIZE,))  # Random timestep for testing
    with pytest.raises(ValueError):
        _, _ = attention(encoder_outputs, method="invalid_method", align_method="dot", timestep=timestep, decoder_hidden=hidden)
        
    
def test_local_attention_method_exception_shapes():
    global first_batch
    
    attention = LocalAttention(hidden_dim=HIDDEN_DIM)
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
    timestep = torch.randint(0, MAX_LENGTH, (BATCH_SIZE,))  # Random timestep for testing
    with pytest.raises(ValueError):
        _, _ = attention(encoder_outputs, method="predictive", align_method="invalid_method", timestep=timestep, decoder_hidden=hidden)

    
def test_data_loader_shapes():
    global train_data_loader
    for batch in train_data_loader:
        src_ids = batch['src_ids']
        trg_ids = batch['trg_ids']
        break
    assert src_ids.shape == (BATCH_SIZE, MAX_LENGTH)
    assert trg_ids.shape == (BATCH_SIZE, MAX_LENGTH)    
    
    
 
@pytest.mark.parametrize("align_method", ["dot", "general", "concat"])
def test_decoder_global_attention_shapes(align_method):
    decoder = Decoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL_DECODER,
        attention_type='global'  
    )
    
    # Inputs and initial states
    input_tensor = first_batch['trg_ids']
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
    method = None
    timestep = None
    
    # Run the decoder
    output, hidden, attn_weights = decoder(
        input=input_tensor, 
        encoder_outputs=encoder_outputs, 
        hidden=hidden, 
        align_method=align_method,
        method=method,
        timestep=timestep
    )

    # Assert output shapes
    assert output.shape == (BATCH_SIZE, MAX_LENGTH, OUTPUT_DIM)
    assert hidden.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert attn_weights.shape == (BATCH_SIZE, MAX_LENGTH)    
    
  
@pytest.mark.parametrize(
    "method, align_method",
    itertools.product(["monotonic", "predictive"], ["dot", "general", "concat"])
)
def test_decoder_local_attention_shapes(method, align_method):
    decoder = Decoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL_DECODER,
        attention_type='local'  
    )
    
    # Inputs and initial states
    input_tensor = first_batch['trg_ids']
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
    timestep = torch.randint(0, MAX_LENGTH, (BATCH_SIZE,))  # Random timestep for testing
    
    # Run the decoder
    output, hidden, attn_weights = decoder(
        input=input_tensor, 
        encoder_outputs=encoder_outputs, 
        hidden=hidden, 
        align_method=align_method,
        method=method,
        timestep=timestep
    )

    # Assert output shapes
    assert output.shape == (BATCH_SIZE, MAX_LENGTH, OUTPUT_DIM)
    assert hidden.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert attn_weights.shape == (BATCH_SIZE, MAX_LENGTH)    