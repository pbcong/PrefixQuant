import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from transformers.cache_utils import DynamicCache


class TrainablePrefixEmbedding(nn.Module):
    """
    Module for trainable prefix embeddings directly for KV cache.
    It creates trainable embeddings for both key and value vectors of the attention mechanism.
    """
    def __init__(
        self,
        num_prefix_tokens: int,
        num_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        device: str = 'cuda'
    ):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_attention_heads
        
        # Create trainable embeddings for each layer's key and value vectors
        # Shape: [num_layers, 2, num_key_value_heads, num_prefix_tokens, head_dim]
        # where dim 1 is for keys (0) and values (1)
        self.prefix_embeddings = nn.Parameter(
            torch.randn(
                num_layers,
                2,  # keys and values
                num_key_value_heads,
                num_prefix_tokens,
                self.head_dim,
            ),
            requires_grad=True  # Make it trainable
        )
        
        # Initialize with small random values
        nn.init.normal_(self.prefix_embeddings, std=0.02)
        
    def forward(self):
        """
        Generate KV cache for the trainable prefix tokens.
        """
        # Create a list to hold the past key values for each layer
        past_key_values = []
        
        # For each layer, extract and format the key and value tensors
        for layer_idx in range(self.num_layers):
            # Get layer-specific key and value embeddings
            # Shape: [num_key_value_heads, num_prefix_tokens, head_dim]
            k_prefix = self.prefix_embeddings[layer_idx, 0]
            v_prefix = self.prefix_embeddings[layer_idx, 1]
            
            # Reshape to batch dimension 1
            # Shape: [1, num_key_value_heads, num_prefix_tokens, head_dim]
            k_prefix = k_prefix.unsqueeze(0)
            v_prefix = v_prefix.unsqueeze(0)
            
            # Add to past_key_values as a tuple (key, value)
            past_key_values.append((k_prefix, v_prefix))
        
        return tuple(past_key_values)

    def to_dynamic_cache(self, batch_size=1):
        """
        Convert to dynamic cache format for inference
        """
        past_key_values = self.forward()
        # Duplicate for batch size if needed
        if batch_size > 1:
            past_key_values = kv_cache_repeat(past_key_values, batch_size)
        
        return DynamicCache.from_legacy_cache(past_key_values)
    
    def get_smoothness_loss(self, weight_smoothness_alpha=0.01, group_size=8):
        """
        Calculate a smoothness loss to make the embedding weights more quantization-friendly.
        
        Args:
            weight_smoothness_alpha: Weight for the smoothness term
            group_size: Group size for calculating group-wise statistics (mimics quantization groups)
            
        Returns:
            loss: A scalar tensor representing the smoothness loss
        """
        # We'll calculate two components:
        # 1. Group-wise variance loss to promote similar values within groups (easier quantization)
        # 2. Global variance loss to promote using the full range of values consistently
        
        # Reshape embedding to calculate group statistics
        # Original shape: [num_layers, 2, num_key_value_heads, num_prefix_tokens, head_dim]
        embeddings = self.prefix_embeddings
        
        # Group-wise variance for each layer/key_value/head grouping
        grouped_emb = embeddings.view(self.num_layers, 2, self.num_key_value_heads, 
                                      self.num_prefix_tokens, -1, group_size)
        
        # Calculate group-wise statistics
        group_mean = grouped_emb.mean(dim=-1, keepdim=True)
        group_var = ((grouped_emb - group_mean) ** 2).mean(dim=-1)
        
        # We want to minimize the group variance (flatter distribution within groups)
        # But maintain reasonable global variance (use the dynamic range)
        group_var_loss = group_var.mean()
        
        # Promote balanced positive/negative values (centered around zero)
        global_mean = embeddings.mean()
        global_mean_loss = global_mean.abs()
        
        # Calculate outlier penalty - large values that deviate too much
        # Penalize values outside 2 standard deviations
        global_std = embeddings.std()
        z_scores = (embeddings - embeddings.mean()) / (global_std + 1e-6)
        outlier_penalty = F.relu(z_scores.abs() - 2.0).mean()
        
        # Combine the losses
        smoothness_loss = weight_smoothness_alpha * (
            0.5 * group_var_loss + 
            0.2 * global_mean_loss + 
            0.3 * outlier_penalty
        )
        
        return smoothness_loss
    
    def get_attention_flatness_loss(self, q_input=None, past_key_values=None, 
                                   attention_flatness_alpha=0.01):
        """
        Calculate loss to encourage flatter attention patterns.
        
        Args:
            q_input: Query tensor from model (optional)
            past_key_values: Generated KV cache (optional, will be generated if None)
            attention_flatness_alpha: Weight for the attention flatness term
            
        Returns:
            loss: A scalar tensor representing the attention flatness loss
        """
        # If no past_key_values provided, generate them
        if past_key_values is None:
            past_key_values = self.forward()
        
        # We can calculate this in two ways:
        # 1. If q_input is provided, calculate actual attention scores and penalize variance
        # 2. If not, directly penalize variance in the key/value vectors
        
        if q_input is not None:
            # Calculate attention scores and penalize non-uniformity
            attn_scores_list = []
            
            for layer_idx, (k, v) in enumerate(past_key_values):
                # Calculate attention scores: Q·K^T / sqrt(d_k)
                # q shape: [batch_size, num_heads, seq_len, head_dim]
                # k shape: [batch_size, num_kv_heads, prefix_tokens, head_dim]
                
                # For simplicity, only calculate with first batch item and first sequence item
                q = q_input[0, :, 0].unsqueeze(0).unsqueeze(2)  # [1, num_heads, 1, head_dim]
                
                # Expand k if needed for multi-query attention
                if self.num_attention_heads > self.num_key_value_heads:
                    # Multi-query attention: repeat k for each query head
                    expand_factor = self.num_attention_heads // self.num_key_value_heads
                    k_expanded = k.repeat_interleave(expand_factor, dim=1)
                else:
                    k_expanded = k
                    
                # Calculate attention scores
                attn_scores = torch.matmul(q, k_expanded.transpose(-1, -2)) / (self.head_dim ** 0.5)
                attn_scores_list.append(attn_scores)
            
            # Stack scores from all layers
            all_attn_scores = torch.cat(attn_scores_list, dim=1)
            
            # Calculate softmax for each head's attention
            attn_probs = F.softmax(all_attn_scores, dim=-1)
            
            # Measure non-uniformity: we want the attention to be spread evenly
            # Perfect uniformity would be 1/num_prefix_tokens for all positions
            uniform_target = torch.ones_like(attn_probs) / self.num_prefix_tokens
            uniformity_loss = F.kl_div(
                attn_probs.log(), 
                uniform_target, 
                reduction='batchmean'
            )
            
            return attention_flatness_alpha * uniformity_loss
        
        else:
            # Without queries, penalize variance directly in the key vectors
            # High variance in keys leads to peaky attention distributions
            key_vectors = torch.cat([k for k, _ in past_key_values], dim=1)
            
            # Calculate variance across the prefix token dimension
            key_mean = key_vectors.mean(dim=2, keepdim=True)
            key_var = ((key_vectors - key_mean) ** 2).mean(dim=2).mean()
            
            return attention_flatness_alpha * key_var


def kv_cache_repeat(past_key_values, batch_size):
    """
    Repeat the KV cache for batch inference.
    Copied from model_utils but included here for completeness.
    """
    if past_key_values is None:
        return None
    
    def _repeat_kv(kv):
        # kv is a tuple of (k, v), each of shape [batch, num_heads, seq_len, head_dim]
        if isinstance(kv, tuple):
            return tuple(x.repeat(batch_size, 1, 1, 1) if x is not None else None for x in kv)
        else:
            return kv.repeat(batch_size, 1, 1, 1) if kv is not None else None
    
    return tuple(_repeat_kv(layer_kv) for layer_kv in past_key_values)


def init_trainable_prefix_embedding(model, num_prefix_tokens, init_from_tokens=None, tokenizer=None):
    """
    Initialize a trainable prefix embedding module.
    
    Args:
        model: The transformer model
        num_prefix_tokens: Number of prefix tokens to create
        init_from_tokens: Optional token IDs to initialize the embeddings from
        tokenizer: Tokenizer needed if init_from_tokens is provided
    
    Returns:
        TrainablePrefixEmbedding module
    """
    config = model.config
    num_layers = len(model.model.layers)
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    
    # Create prefix embedding module
    prefix_embedding = TrainablePrefixEmbedding(
        num_prefix_tokens=num_prefix_tokens,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_size=hidden_size,
        device=next(model.parameters()).device
    )
    
    # Initialize from existing tokens if specified
    if init_from_tokens is not None and tokenizer is not None:
        # Process the tokens through the model to get KV cache
        input_ids = torch.tensor([init_from_tokens], device=next(model.parameters()).device)
        with torch.no_grad():
            outputs = model(input_ids, return_dict=True, use_cache=True)
            past_key_values = outputs.past_key_values
        
        # Copy values to our trainable embeddings
        for layer_idx, (k, v) in enumerate(past_key_values):
            # k and v have shape [batch_size, num_heads, seq_len, head_dim]
            prefix_embedding.prefix_embeddings.data[layer_idx, 0] = k[0]
            prefix_embedding.prefix_embeddings.data[layer_idx, 1] = v[0]
    
    return prefix_embedding
 

def compute_combined_loss(task_loss, prefix_embedding, q_input=None, 
                         weight_smoothness_alpha=0.01, 
                         attention_flatness_alpha=0.01):
    """
    Compute combined loss with regularization for quantization-friendly prefix.
    
    Args:
        task_loss: Primary task loss (e.g., MSE between model outputs)
        prefix_embedding: TrainablePrefixEmbedding instance
        q_input: Query input tensors (optional)
        weight_smoothness_alpha: Weight for embedding smoothness term
        attention_flatness_alpha: Weight for attention flatness term
        
    Returns:
        total_loss: Combined loss value
    """
    # Calculate regularization losses
    smoothness_loss = prefix_embedding.get_smoothness_loss(
        weight_smoothness_alpha=weight_smoothness_alpha
    )
    
    flatness_loss = prefix_embedding.get_attention_flatness_loss(
        q_input=q_input,
        attention_flatness_alpha=attention_flatness_alpha
    )
    
    # Combine losses
    total_loss = task_loss + smoothness_loss + flatness_loss
    
    return total_loss, {
        'task_loss': task_loss.item(),
        'smoothness_loss': smoothness_loss.item(),
        'flatness_loss': flatness_loss.item()
    }