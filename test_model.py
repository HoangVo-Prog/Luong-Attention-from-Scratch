import torch
import test_model
from model import *
from config import *
from attention import *
import pytest

for batch in train_data_loader:
    first_batch = batch
    break  # Just to check the first batch


def test_encoder_output_shapes():
    global first_batch
    
    encoder = Encoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL_ENCODER
    )
    input_tensor = first_batch['src_ids']
    output, hidden = encoder(input_tensor)
    assert output.shape == (BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)
    assert hidden.shape == (BATCH_SIZE, HIDDEN_DIM)


def test_global_attention_dot_shapes():
    global first_batch
    
    attention = GlobalAttention(hidden_dim=HIDDEN_DIM)
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
    context, attn_weights = attention(encoder_outputs, hidden, method="dot")
    assert context.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert attn_weights.shape == (BATCH_SIZE, MAX_LENGTH)
    
    
def test_global_attention_general_shapes():
    global first_batch
    
    attention = GlobalAttention(hidden_dim=HIDDEN_DIM)
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
    context, attn_weights = attention(encoder_outputs, hidden, method="general")
    assert context.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert attn_weights.shape == (BATCH_SIZE, MAX_LENGTH)
    
    
def test_global_attention_concat_shapes():
    global first_batch
    
    attention = GlobalAttention(hidden_dim=HIDDEN_DIM)
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
    context, attn_weights = attention(encoder_outputs, hidden, method="concat")
    assert context.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert attn_weights.shape == (BATCH_SIZE, MAX_LENGTH)


def test_local_attention_monotonic_dot_shapes():
    global first_batch
    
    attention = LocalAttention(hidden_dim=HIDDEN_DIM)
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
    timestep = torch.randint(0, MAX_LENGTH, (BATCH_SIZE,))  # Random timestep for testing
    context, attn_weights = attention(encoder_outputs, method="monotonic", align_method="dot", timestep=timestep, decoder_hidden=hidden)
    assert context.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert attn_weights.shape == (BATCH_SIZE, MAX_LENGTH)
    
    
def test_local_attention_monotonic_general_shapes():
    global first_batch
    
    attention = LocalAttention(hidden_dim=HIDDEN_DIM)
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
    timestep = torch.randint(0, MAX_LENGTH, (BATCH_SIZE,))  # Random timestep for testing
    context, attn_weights = attention(encoder_outputs, method="monotonic", align_method="general", timestep=timestep, decoder_hidden=hidden)
    assert context.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert attn_weights.shape == (BATCH_SIZE, MAX_LENGTH)
    

def test_local_attention_monotonic_concat_shapes():
    global first_batch
    
    attention = LocalAttention(hidden_dim=HIDDEN_DIM)
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
    timestep = torch.randint(0, MAX_LENGTH, (BATCH_SIZE,))  # Random timestep for testing
    context, attn_weights = attention(encoder_outputs, method="monotonic", align_method="concat", timestep=timestep, decoder_hidden=hidden)
    assert context.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert attn_weights.shape == (BATCH_SIZE, MAX_LENGTH)
    
    
def test_local_attention_predictive_dot_shapes():
    global first_batch
    
    attention = LocalAttention(hidden_dim=HIDDEN_DIM)
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
    timestep = torch.randint(0, MAX_LENGTH, (BATCH_SIZE,))  # Random timestep for testing
    context, attn_weights = attention(encoder_outputs, method="predictive", align_method="dot", timestep=timestep, decoder_hidden=hidden)
    assert context.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert attn_weights.shape == (BATCH_SIZE, MAX_LENGTH)
    

def test_local_attention_predictive_general_shapes():
    global first_batch
    
    attention = LocalAttention(hidden_dim=HIDDEN_DIM)
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
    timestep = torch.randint(0, MAX_LENGTH, (BATCH_SIZE,))  # Random timestep for testing
    context, attn_weights = attention(encoder_outputs, method="predictive", align_method="general", timestep=timestep, decoder_hidden=hidden)
    assert context.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert attn_weights.shape == (BATCH_SIZE, MAX_LENGTH)
    
    
def test_local_attention_predictive_concat_shapes():
    global first_batch
    
    attention = LocalAttention(hidden_dim=HIDDEN_DIM)
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
    timestep = torch.randint(0, MAX_LENGTH, (BATCH_SIZE,))  # Random timestep for testing
    context, attn_weights = attention(encoder_outputs, method="predictive", align_method="concat", timestep=timestep, decoder_hidden=hidden)
    assert context.shape == (BATCH_SIZE, HIDDEN_DIM)
    assert attn_weights.shape == (BATCH_SIZE, MAX_LENGTH)   
    
    
def test_global_attention_exception_shapes():
    global first_batch
    
    attention = GlobalAttention(hidden_dim=HIDDEN_DIM)
    encoder_outputs = torch.randn(BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM)  # Encoder output
    hidden = torch.randn(BATCH_SIZE, HIDDEN_DIM)  # Decoder hidden state
    with pytest.raises(ValueError):
        _, _ = attention(encoder_outputs, hidden, method="invalid_method")
        
        
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
    
    
