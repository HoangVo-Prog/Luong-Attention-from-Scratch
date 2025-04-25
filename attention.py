import torch
import torch.nn as nn
from config import *


class AttentionBase(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionBase, self).__init__()
        self.Va = nn.Linear(hidden_dim * 2, 1)
        self.Wa = nn.Linear(hidden_dim, hidden_dim)

    def align(self, decoder_hidden, encoder_outputs, method):
        if method == "dot":
            return torch.bmm(decoder_hidden.unsqueeze(1), encoder_outputs.transpose(1, 2)).squeeze(1)
        elif method == "general":
            return torch.bmm(decoder_hidden.unsqueeze(1), self.Wa(encoder_outputs).transpose(1, 2)).squeeze(1)
        elif method == "concat":
            decoder_hidden_expanded = self.Wa(decoder_hidden).unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
            concat_input = torch.cat([decoder_hidden_expanded, encoder_outputs], dim=-1)
            return self.Va(torch.tanh(concat_input)).squeeze(2)
        else:
            print("Current attention method: ", method)
            raise ValueError("Invalid attention method. Choose 'dot', 'general', or 'concat'.")


class GlobalAttention(AttentionBase):
    def __init__(self, hidden_dim):
        super(GlobalAttention, self).__init__(hidden_dim)

    def forward(self, encoder_outputs, decoder_hidden, align_method, method=None, timestep=None):
        attention_weights = torch.softmax(self.align(decoder_hidden, encoder_outputs, align_method), dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context_vector, attention_weights
        
           
class LocalAttention(AttentionBase):
    def __init__(self, hidden_dim, sigma=1):
        super(LocalAttention, self).__init__(hidden_dim)
        
        self.sigma = sigma
        self.Vp = nn.Linear(hidden_dim, 1)
        
    def monotonic_attention(self, timestep):
        return timestep
    
    def predictive_attention(self, encoder_outputs):
        return MAX_LENGTH*self.Vp(torch.softmax(self.Wa(encoder_outputs), dim=1))  # [batch_size, seq_len, 1]
    
    
    def gaussian_term(self, s, p_t):
        # s is a tensor of shape [1, seq_len] representing sequence positions.
        # p_t is the predicted position tensor:
        #   - For monotonic attention: p_t is of shape [batch_size, 1] (a single predicted position per batch element).
        #   - For predictive attention: p_t is of shape [batch_size, seq_len, 1] (predicted positions for each timestep in the sequence).
        
        if p_t.dim() == 3:  # For predictive attention where p_t is of shape [batch_size, seq_len, 1]
            p_t = p_t.squeeze(-1)  # Squeeze the last dimension to get p_t shape [batch_size, seq_len]
        
        if p_t.dim() == 1:  # For monotonic attention where p_t is of shape [batch_size, 1]
            p_t = p_t.unsqueeze(1)  # Reshape p_t to [batch_size, 1]
            p_t = p_t.expand(-1, s.size(1))  # Broadcast p_t across seq_len to match shape [batch_size, seq_len]
        
        s = s.expand(p_t.size(0), -1)  # Broadcast s (of shape [1, seq_len]) to [batch_size, seq_len] for element-wise comparison.
        
        # Compute the Gaussian term element-wise: exp(-((s - p_t) ** 2) / (2 * self.sigma ** 2))
        # This computes the Gaussian attention weight for each position in the sequence.
        return torch.exp(-((s - p_t) ** 2) / (2 * self.sigma ** 2)) # [batch_size, seq_len]


    
    def forward(self, encoder_outputs, decoder_hidden, align_method, method, timestep):
        if method == 'monotonic':
            p = self.monotonic_attention(timestep)
        elif method == 'predictive':
            p = self.predictive_attention(encoder_outputs)
        else:
            raise ValueError("Invalid attention method. Choose 'monotonic' or 'predictive'.")

        # s = torch.arange(MAX_LENGTH).float().unsqueeze(0).to(encoder_outputs.DEVICE)
        s = torch.arange(MAX_LENGTH).float().unsqueeze(0).to(encoder_outputs)

        gaussian_scores = self.gaussian_term(s, p)
        attention_weights = torch.softmax(self.align(decoder_hidden, encoder_outputs, align_method), dim=1) * gaussian_scores
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context_vector, attention_weights
    

        