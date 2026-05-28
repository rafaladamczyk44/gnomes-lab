"""
Masked Attention Implementation in Pure Python (No PyTorch)

This module implements a masked self-attention mechanism using NumPy as the
only dependency. It supports:
- Causal masking (for autoregressive models like LLMs)
- Custom masks for specific attention patterns
"""

import numpy as np


class MaskedAttention:
    """
    A masked self-attention layer implemented in pure Python/NumPy.
    
    This is the building block for Transformer models, with masking support
    to prevent positions from attending to subsequent positions (causal mask).
    
    Parameters:
        d_model (int): Dimension of model embeddings
        num_heads (int): Number of attention heads
        dropout_rate (float, optional): Dropout probability for training
    
    Attributes:
        q_weight, k_weight, v_weight: Linear projection weights (d_model -> d_head)
        output_weight: Output projection weight (num_heads * d_head -> d_model)
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # Compute head dimension
        d_head = d_model // num_heads
        
        # Initialize weights with Xavier initialization
        self.q_weight = self._init_weights(d_model, d_head)
        self.k_weight = self._init_weights(d_model, d_head)
        self.v_weight = self._init_weights(d_model, d_head)
        
        # Output projection weights
        out_dim = num_heads * d_head
        self.out_weight = self._init_weights(out_dim, d_model)
    
    def _init_weights(self, in_features: int, out_features: int):
        """Xavier/Glorot uniform initialization."""
        scale = np.sqrt(1.0 / (in_features + out_features))
        return np.random.randn(in_features, out_features).astype(np.float32) * scale
    
    def _apply_mask(self, logits: np.ndarray, mask: np.ndarray = None):
        """Apply attention mask to logits."""
        return self._masked_softmax(logits, mask)
    
    def compute_masked_attention(self, x: np.ndarray, mask: np.ndarray = None):
        """
        Compute masked self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask of shape (seq_len, seq_len) or 
                  (batch_size, 1, seq_len, seq_len). If None, applies causal mask.
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply causal mask if no custom mask provided
        if mask is None:
            mask = self._create_causal_mask(seq_len)
        
        # Compute Q, K, V projections - each has shape (batch_size, seq_len, d_head)
        q = np.asarray(x @ self.q_weight.T).astype(np.float32)  # (B, S, d_head)
        k = np.asarray(x @ self.k_weight.T).astype(np.float32)  # (B, S, d_head)
        v = np.asarray(x @ self.v_weight.T).astype(np.float32)  # (B, S, d_head)
        
        # Repeat for all heads: (B, S, d_head) -> (B, num_heads * S, d_head)
        q = np.repeat(q[np.newaxis, :, :].astype(np.float64), self.num_heads, axis=0).astype(np.float32)
        k = np.repeat(k[np.newaxis, :, :].astype(np.float64), self.num_heads, axis=0).astype(np.float32)
        v = np.repeat(v[np.newaxis, :, :].astype(np.float64), self.num_heads, axis=0).astype(np.float32)
        
        # Reshape for multi-head: (B, H*S, d_head) -> (B, H, S, d_head)
        q = self._reshape_for_heads(q)
        k = self._reshape_for_heads(k)
        v = self._reshape_for_heads(v)
        
        # Compute attention logits: (B, H, S_q, d_head) @ (B, H, S_k, d_head)^T
        # Result: (B, H, S_q, S_k)
        # Need to transpose k's last dimension for proper head-wise matrix mult
        attention_logits = q @ np.transpose(k, (0, 1, 3, 2)) / np.sqrt(self.d_head)
        
        # Apply mask and compute softmax
        attention_probs = self._apply_mask(attention_logits, mask)
        
        # Compute weighted sum over V
        output = attention_probs @ v
        
        # Reshape back: (B, H, S_q, d_head) -> (B, S_q, H*d_head)
        output = self._reshape_output(output)
        
        # Apply output projection
        output = output @ self.out_weight
        
        return output
    
    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """
        Create causal mask for autoregressive models.
        
        Positions can only attend to themselves and previous positions.
        
        Args:
            seq_len: Sequence length
        
        Returns:
            Causal mask of shape (seq_len, seq_len) with -inf for masked positions
        """
        mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32))
        mask = (mask == 1).astype(np.float32)
        
        # Apply negative infinity to masked positions (future tokens)
        mask = np.where(mask == 0, -np.inf, 1.0)
        
        return mask
    
    def _create_bidirectional_mask(self, seq_len: int) -> np.ndarray:
        """
        Create bidirectional mask (no masking).
        
        All positions can attend to all other positions.
        """
        return np.ones((seq_len, seq_len), dtype=np.float32)
    
    def _create_padding_mask(self, lengths: np.ndarray, max_len: int) -> np.ndarray:
        """
        Create padding mask from sequence lengths.
        
        Positions to the right of padding positions are masked.
        
        Args:
            lengths: Array of shape (batch_size,) containing sequence length for each sample
            max_len: Maximum sequence length in the batch
        
        Returns:
            Padding mask of shape (batch_size, 1, max_len, max_len)
        """
        mask = np.ones((len(lengths), 1, max_len, max_len), dtype=np.float32)
        
        for i, length in enumerate(lengths):
            mask[i, 0, :, i] = np.triu(np.ones((max_len,), dtype=np.float32))[:length].astype(np.float32)
        
        # Apply negative infinity to padding positions (right of actual sequence)
        mask = np.where(mask == 0, -np.inf, 1.0)
        
        return mask
    
    def _masked_softmax(self, logits: np.ndarray, mask: np.ndarray = None):
        """
        Apply masking to attention logits and compute softmax.
        
        Args:
            logits: Attention logits of shape (*, seq_len, seq_len)
            mask: Optional mask of same shape as logits
        
        Returns:
            Softmax probabilities with masking applied
        """
        # Apply mask (add to logits)
        if mask is not None:
            logits = logits + mask
        
        # Create shape for broadcasting
        batch_shape = list(logits.shape[:-2]) if len(logits.shape) > 2 else []
        
        # Transpose for softmax over last two dimensions
        logits = logits.reshape(-1, logits.shape[-2], logits.shape[-1])
        
        # Compute log softmax for numerical stability
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        softmax_probs = exp_logits / np.sum(exp_logits, axis=-1)
        
        # Mask out padded positions (set to -inf before softmax if needed)
        return softmax_probs.reshape(batch_shape + [logits.shape[-2], logits.shape[-1]])
    
    def _reshape_for_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape tensor for multi-head attention."""
        batch_size, seq_len, d_model = x.shape
        d_head = self.d_model // self.num_heads
        
        # (B, S, H*d) -> (B, H, S, d)
        return x.reshape(batch_size, self.num_heads, seq_len, d_head)
    
    def _reshape_output(self, x: np.ndarray) -> np.ndarray:
        """Reshape output from multi-head attention."""
        batch_size, num_heads, seq_len, d_head = x.shape
        
        # (B, H, S, d) -> (B, S, H*d)
        return x.reshape(batch_size, seq_len, num_heads * d_head)


class MultiHeadAttention:
    """
    Multi-head attention with separate Q, K, V projections.
    
    This is the full attention head implementation with optional masking.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.d_head = d_model // num_heads
        
        # Initialize weights
        scale = np.sqrt(1.0 / (d_model + self.d_head))
        self.q_weight = np.random.randn(d_model, self.d_head).astype(np.float32) * scale
        self.k_weight = np.random.randn(d_model, self.d_head).astype(np.float32) * scale
        self.v_weight = np.random.randn(d_model, self.d_head).astype(np.float32) * scale
        
        # Output projection
        self.out_weight = np.random.randn(num_heads * self.d_head, d_model).astype(np.float32)
    
    def forward(self, x: np.ndarray, mask: np.ndarray = None):
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Project to Q, K, V
        q = x @ self.q_weight.transpose(0, 1)
        k = x @ self.k_weight.transpose(0, 1)
        v = x @ self.v_weight.transpose(0, 1)
        
        # Compute attention scores
        attn_scores = q @ k.transpose(-2, -1) / np.sqrt(self.d_head)
        
        # Apply mask
        if mask is not None:
            attn_scores = np.where(mask, -np.inf, attn_scores)
        
        # Softmax
        attn_probs = np.softmax(attn_scores, axis=-1)
        
        # Weighted sum over V
        output = attn_probs @ v
        
        return output
    
    def forward_with_output_projection(self, x: np.ndarray, mask: np.ndarray = None):
        """
        Forward pass with output projection.
        
        Returns:
            (output, attention_weights) tuple for inspection/debugging
        """
        output = self.forward(x, mask)
        
        # Apply output projection
        projected_output = x @ self.out_weight
        
        return projected_output


# Example usage and testing
if __name__ == "__main__":
    # Create a simple masked attention model
    d_model = 128
    num_heads = 4
    
    attention = MaskedAttention(d_model, num_heads)
    
    # Example input: batch of 2 sequences with length 5 and d_model=128
    np.random.seed(42)
    x = np.random.randn(2, 5, d_model).astype(np.float32)
    
    # Compute causal masked attention (default behavior for autoregressive models)
    print("Computing causal masked attention...")
    output = attention.compute_masked_attention(x, mask=None)  # Causal mask applied by default
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should be (2, 5, 128)
    print(f"Attention is causal: positions can only attend to themselves and previous positions")
    
    # Compute with custom bidirectional mask (for encoder models)
    print("\nComputing bidirectional attention...")
    bidirectional_mask = attention.create_bidirectional_mask(5)
    output_bi = attention.compute_masked_attention(x, mask=bidirectional_mask)
    print(f"Output shape: {output_bi.shape}")
    
    # Compute with padding mask (for variable length sequences)
    print("\nComputing attention with padding mask...")
    lengths = np.array([4, 5])  # Two sequences of different lengths
    padding_mask = attention.create_padding_mask(lengths, max_len=5)
    output_pad = attention.compute_masked_attention(x, mask=padding_mask)
    print(f"Output shape: {output_pad.shape}")