import torch
from torch import nn
from config import *
from attention import *
from Data.data import cache_or_process



torch.manual_seed(SEED)

train_data_loader, valid_data_loader, test_data_loader, en_tokenizer, vi_tokenizer = cache_or_process()

INPUT_DIM = en_tokenizer.get_vocab_size()
OUTPUT_DIM = vi_tokenizer.get_vocab_size()

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, output_dim, num_layers, dropout, bidirectional):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2 if bidirectional else hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
       
        # x = x.to(DEVICE)
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, _) = self.lstm(embedded)
        
        outputs = self.layer_norm(outputs)  
        
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        outputs = self.fc(outputs)
        hidden = self.fc_hidden(hidden)
        outputs = self.dropout(outputs)
        
        return outputs, hidden # (BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM), (BATCH_SIZE, HIDDEN_DIM)

    
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, output_dim, num_layers, dropout, bidirectional, attention_type):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)

        if attention_type == 'global':
            self.attention = GlobalAttention(hidden_dim)
        elif attention_type == 'local':
            self.attention = LocalAttention(hidden_dim)
        else:
            raise ValueError("Invalid attention type. Choose 'global' or 'local'.")

        self.fc_hidden = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2 if bidirectional else hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 4 if bidirectional else hidden_dim*2, hidden_dim)
        
    def forward(self, input, encoder_outputs, hidden, align_method, method, timestep):
        # input.shape: (BATCH_SIZE, 1), encoder_outputs.shape: (BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM), hidden.shape: (BATCH_SIZE, HIDDEN_DIM)
        # input = input.to(DEVICE)
        embedded = self.dropout(self.embedding(input))
        
        # cell = torch.zeros_like(hidden).to(DEVICE)
        hidden = hidden.repeat(self.lstm.num_layers * 2 if self.lstm.bidirectional else 1, 1, 1)
        
        cell = torch.zeros_like(hidden)
        outputs, (hidden, _) = self.lstm(embedded, (hidden, cell)) # [batch_size, seq_length, hidden_size]
        print("after lstm, outputs shape, hidden shape:", outputs.shape, hidden.shape)
        outputs = self.layer_norm(outputs)
        
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        context_vector, attn_weights = self.attention(encoder_outputs=encoder_outputs, 
                                               decoder_hidden=hidden, 
                                               align_method=align_method, 
                                               method=method,
                                               timestep=timestep)  # [batch_size, hidden_dim], [batch_size, seq_len]
        
        outputs = torch.cat((outputs, context_vector.unsqueeze(1).repeat(1, outputs.size(1), 1)), dim=-1)

        hidden = self.fc_hidden(hidden).squeeze(0)
        outputs = self.fc(outputs)
        outputs = self.dropout(outputs)
        
        predictions = self.fc_out(outputs.squeeze(1))
        
        return predictions, hidden, attn_weights
    
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        outputs = torch.zeros(batch_size, trg_len, OUTPUT_DIM).to(DEVICE)
        
        encoder_outputs, hidden = self.encoder(src)

        input = trg[:, 0]  # trg.shape: (BATCH_SIZE, MAX_LENGTH), input: (BATCH_SIZE, 1)
        
        for t in range(1, MAX_LENGTH):
            output, hidden, _ = self.decoder(
                input.unsqueeze(1), encoder_outputs, hidden
            )

            outputs[:, t, :] = output
            
            teacher_force = torch.rand(1, device=DEVICE).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs  
