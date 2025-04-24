import torch
from torch import nn
from config import *
from attention import BahdanauAttention
from Data.data import cache_or_process



torch.manual_seed(SEED)

train_data_loader, valid_data_loader, test_data_loader, en_tokenizer, vi_tokenizer = cache_or_process()

INPUT_DIM = en_tokenizer.get_vocab_size()
OUTPUT_DIM = vi_tokenizer.get_vocab_size()

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, bidirectional):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2 if bidirectional else hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
       
        x = x.to(DEVICE)
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.rnn(embedded)
        
        outputs = self.batch_norm(outputs.permute(0, 2, 1))  
        outputs = outputs.permute(0, 2, 1) 
        
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        outputs = self.fc(outputs)
        hidden = self.fc_hidden(hidden)
        
        return outputs, hidden # (BATCH_SIZE, MAX_LENGTH, HIDDEN_DIM), (4, BATCH_SIZE, HIDDEN_DIM)

    
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, bidirectional):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.rnn = nn.RNN(hidden_dim*2, hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.attention = BahdanauAttention(hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2 if bidirectional else hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim)
        
    def forward(self, input, encoder_outputs, hidden):
        
        input = input.to(DEVICE)
        embedded = self.dropout(self.embedding(input))
        context, attn_weights = self.attention(hidden, encoder_outputs)
        
        rnn_input = torch.cat((embedded, context), dim=2)
        outputs, hidden = self.rnn(rnn_input, hidden.unsqueeze(0).repeat(self.rnn.num_layers*(int(self.rnn.bidirectional)+1), 1, 1))
        
        outputs = self.batch_norm(outputs.permute(0, 2, 1))
        outputs = outputs.permute(0, 2, 1)
        
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            
        hidden = self.fc_hidden(hidden)
        outputs = self.fc(outputs)
        predictions = self.fc_out(outputs.squeeze(1))
        
        return predictions, hidden, attn_weights
    
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        
        # outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)
        outputs = torch.zeros(BATCH_SIZE, MAX_LENGTH, OUTPUT_DIM)
        
        encoder_outputs, hidden = self.encoder(src)

        input = trg[:, 0]  # trg.shape: (BATCH_SIZE, MAX_LENGTH), input: (BATCH_SIZE, 1)
        
        for t in range(1, MAX_LENGTH):
            output, hidden, _ = self.decoder(input.unsqueeze(1), encoder_outputs, hidden)
            outputs[:, t] = output
            
            top1 = output.argmax(1)  
            
            input = top1 if torch.rand(1) > teacher_forcing_ratio else trg[:, t]
            
        return outputs  
