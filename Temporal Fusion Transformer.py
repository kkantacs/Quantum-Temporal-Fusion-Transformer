import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLU, self).__init__()
        self.linear_gate = nn.Linear(input_dim, output_dim)
        self.linear_feature = nn.Linear(input_dim, output_dim)

    def forward(self, eta2):
        gate = torch.sigmoid(self.linear_gate(eta2))
        feature = self.linear_feature(eta2)
        eta3 = gate * feature
        return eta3

class GRN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRN, self).__init__()
        self.input_dim = input_dim
        self.linear_a = nn.Linear(input_dim, hidden_dim)
        self.linear_eta2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GLU(hidden_dim, input_dim)

    def forward(self, a, c=None):
        if c is None:
            c = torch.zeros_like(a)
        eta1 = F.elu(self.linear_a(a))
        eta2 = self.linear_eta2(eta1)
        eta3 = self.glu(eta2)
        residual = a + eta3
        output = residual
        return output

class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_vars, d_model, hidden_dim):
        super(VariableSelectionNetwork, self).__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        self.selection_grn = GRN(input_dim=num_vars * d_model, hidden_dim=hidden_dim)
        self.variable_grn = GRN(input_dim=d_model, hidden_dim=hidden_dim)

    def forward(self, xi_t, context=None):
        B, m_x, d_model = xi_t.shape
        assert m_x == self.num_vars and d_model == self.d_model
        xi_flat = xi_t.view(B, -1)
        selection_logits = self.selection_grn(xi_flat, context)
        v_weights = F.softmax(selection_logits.view(B, m_x), dim=-1)
        xi_reshaped = xi_t.reshape(B * m_x, d_model)
        xi_proc = self.variable_grn(xi_reshaped)
        xi_proc = xi_proc.view(B, m_x, d_model)
        v_weights = v_weights.unsqueeze(-1)
        output = torch.sum(v_weights * xi_proc, dim=1)
        return output, v_weights

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model):
        super(SingleHeadAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attn_weights = self.softmax(scores)
        attn_output = torch.matmul(attn_weights, V)
        return attn_output, attn_weights

class MyModel(nn.Module):
    def __init__(self, num_vars_k, num_vars_T, d_model=1, hidden_dim=2, lstm_hidden=1):
        super(MyModel, self).__init__()
        self.vsn_k = VariableSelectionNetwork(num_vars=num_vars_k, d_model=d_model, hidden_dim=hidden_dim)
        self.vsn_T = VariableSelectionNetwork(num_vars=num_vars_T, d_model=d_model, hidden_dim=hidden_dim)
        self.lstm_cell = nn.LSTMCell(input_size=1, hidden_size=lstm_hidden)
        self.glu1 = GLU(input_dim=1, output_dim=1)
        self.grn1 = GRN(input_dim=1, hidden_dim=1)
        self.attn = SingleHeadAttention(d_model=1)
        self.grn2 = GRN(input_dim=1, hidden_dim=1)
        self.glu2 = GLU(input_dim=1, output_dim=1)

    def forward(self, x_k, x_T):
        B, k, num_vars_k = x_k.shape
        _, T, num_vars_T = x_T.shape
        vsn_outputs = []
        for t in range(k):
            xi = x_k[:, t, :].view(B, num_vars_k, 1)
            out, _ = self.vsn_k(xi)
            vsn_outputs.append(out)
        for t in range(T):
            xi = x_T[:, t, :].view(B, num_vars_T, 1)
            out, _ = self.vsn_T(xi)
            vsn_outputs.append(out)
        vsn_seq = torch.stack(vsn_outputs, dim=1)
        h_t = torch.zeros(B, 1)
        c_t = torch.zeros(B, 1)
        lstm_outs = []
        for t in range(k + T):
            x_t = vsn_seq[:, t, :]
            h_t, c_t = self.lstm_cell(x_t, (h_t, c_t))
            lstm_outs.append(h_t)
        lstm_seq = torch.stack(lstm_outs, dim=1)
        combined1 = self.glu1(lstm_seq) + vsn_seq
        grn_out = self.grn1(combined1)
        attn_out, _ = self.attn(grn_out)
        combined2 = self.glu1(attn_out) + grn_out
        grn_out2 = self.grn2(combined2)
        final = self.glu2(grn_out2) + combined1
        return final[:, -T:, :]
