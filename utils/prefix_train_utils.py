from utils.stat_utils import get_prefixed_tokens
from utils.data_utils import get_loaders
from utils.quant_utils import get_act_stat
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#wrap the model with prefix tokens and make the prefix tokens trainable, train with the loss and update the prefix tokens

class ModelWithPrefix(nn.Module):
    def __init__(self, model, num_prefix_tokens):
        super().__init__()
        self.model = model
        self.num_prefix_tokens = num_prefix_tokens
        self.model_device = next(model.parameters()).device
        self.model_dtype = next(model.parameters()).dtype
        hidden_size = model.config.hidden_size
        
        # Initialize with much smaller scale
        self.prefix_embeddings = nn.Parameter(
            torch.randn(num_prefix_tokens, hidden_size, device=self.model_device, dtype=torch.float32) * 0.01,
            requires_grad=True
        )
        
        # Stability parameters
        self.eps = 1e-5
        self.max_norm = 1e2
        self.min_norm = 1e-5
        self.grad_scale = 0.1
        
        # Layer norm without affine transform for stability
        self.prefix_norm = nn.LayerNorm(hidden_size, eps=self.eps, elementwise_affine=False)
        
    def set_prefix_embeddings(self, prefix_embeddings):
        if isinstance(prefix_embeddings, torch.Tensor):
            assert prefix_embeddings.shape == self.prefix_embeddings.shape, \
                f"Shape mismatch: expected {self.prefix_embeddings.shape}, got {prefix_embeddings.shape}"
            self.prefix_embeddings.data = prefix_embeddings.clone().detach().to(device=self.model_device, dtype=self.model_dtype)
        elif isinstance(prefix_embeddings, (list, tuple)):
            token_ids = torch.tensor(prefix_embeddings[:self.num_prefix_tokens], dtype=torch.long, device=self.model_device)
            with torch.no_grad():
                embed_layer = self.model.get_input_embeddings()
                new_embeddings = embed_layer(token_ids)
                self.prefix_embeddings.data = new_embeddings.clone().to(dtype=self.model_dtype)
        else:
            raise ValueError("prefix_embeddings must be a tensor or list of token IDs")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Get input embeddings
        input_embeddings = self.model.get_input_embeddings()(input_ids)
        if torch.isnan(input_embeddings).any():
            raise ValueError("NaN detected in input embeddings")
        
        # Scale gradients during forward pass
        with torch.set_grad_enabled(self.prefix_embeddings.requires_grad):
            if self.prefix_embeddings.requires_grad:
                prefix_emb = self.prefix_embeddings * self.grad_scale
            else:
                prefix_emb = self.prefix_embeddings
            
            # Normalize with stability
            normalized_prefix = self.prefix_norm(prefix_emb)
            norm = torch.norm(normalized_prefix, p=2, dim=1, keepdim=True)
            norm = torch.clamp(norm, min=self.min_norm, max=self.max_norm)
            normalized_prefix = normalized_prefix / norm
        
        # Expand and concatenate
        expanded_prefix = normalized_prefix.unsqueeze(0).expand(batch_size, -1, -1)
        if torch.isnan(expanded_prefix).any():
            raise ValueError("NaN detected in prefix embeddings")
        
        full_embeddings = torch.cat([expanded_prefix, input_embeddings], dim=1)
        if torch.isnan(full_embeddings).any():
            raise ValueError("NaN detected in concatenated embeddings")
        
        # Handle attention mask
        if attention_mask is not None:
            prefix_attention = torch.ones(batch_size, self.num_prefix_tokens, device=device, dtype=attention_mask.dtype)
            full_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
        else:
            seq_len = input_ids.size(1) + self.num_prefix_tokens
            full_attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Forward pass
        try:
            outputs = self.model(
                inputs_embeds=full_embeddings,
                attention_mask=full_attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs
            )
            
            if hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states
                for i, hidden_state in enumerate(hidden_states):
                    if torch.isnan(hidden_state).any():
                        raise ValueError(f"NaN detected in hidden state {i}")
            else:
                raise ValueError("Model outputs do not contain hidden states")
            
            return outputs
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shape: {input_ids.shape}")
            print(f"Prefix embeddings stats: mean={self.prefix_embeddings.mean().item()}, std={self.prefix_embeddings.std().item()}")
            print(f"Full embeddings stats: mean={full_embeddings.mean().item()}, std={full_embeddings.std().item()}")
            raise

def get_outlier_reduction_loss_fn(prefix_len, outlier_threshold=64, target_layers=None):
    """Create loss function that makes activations flat (reduces outliers) for better quantization."""
    def loss_fn(hidden_states_list, prefix_embeddings):
        device = prefix_embeddings.device
        dtype = prefix_embeddings.dtype
        
        total_loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        layer_count = 0
        
        # Process hidden states from multiple layers
        for layer_idx, hidden_states in enumerate(hidden_states_list):
            if target_layers is not None and layer_idx not in target_layers:
                continue
                
            if not hidden_states.requires_grad:
                continue
                
            # Extract non-prefix hidden states
            non_prefix_states = hidden_states[:, prefix_len:, :]
            
            # Flatten to get all activation values
            flat_activations = non_prefix_states.flatten()
            
            # Method 1: Minimize standard deviation to make activations flat
            std_loss = torch.std(flat_activations)
            
            # Method 2: Minimize the difference between max and min (range)
            max_val = torch.max(flat_activations)
            min_val = torch.min(flat_activations)
            range_loss = max_val - min_val
            
            # Method 3: Minimize top percentile values (directly target outliers)
            abs_activations = torch.abs(flat_activations)
            k = max(1, int(0.05 * abs_activations.numel()))  # Top 5% of values
            top_k_values, _ = torch.topk(abs_activations, k)
            outlier_loss = top_k_values.mean()
            
            # Combine the flatness objectives
            layer_loss = 0.4 * std_loss + 0.3 * range_loss + 0.3 * outlier_loss
            
            total_loss = total_loss + layer_loss
            layer_count += 1
        
        if layer_count == 0:
            # Fallback: just regularize prefix embeddings
            return prefix_embeddings.pow(2).mean()
        
        # Average across layers and add small prefix regularization
        avg_loss = total_loss / layer_count
        prefix_reg = 0.01 * prefix_embeddings.pow(2).mean()
        
        final_loss = avg_loss + prefix_reg
        
        return final_loss, avg_loss, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    
    return loss_fn

def train_soft_prefix(model, tokenizer, args, logger):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    logger.info(f"[SoftPrefix] Training on device: {device} with dtype: {dtype}")
    
    model_with_prefix = ModelWithPrefix(model, args.prefix_len)
    model_with_prefix = model_with_prefix.to(device)
    
    # Initialize prefix embeddings
    with torch.no_grad():
        hidden_size = model.config.hidden_size
        scale = 0.01 * math.sqrt(2.0 / (args.prefix_len + hidden_size))
        model_with_prefix.prefix_embeddings.data.normal_(mean=0.0, std=scale)
    
    model_with_prefix.prefix_embeddings.requires_grad_(True)
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    
    # Create data loader
    train_loader, _ = get_loaders(
        args.calib_dataset, tokenizer,
        train_size=args.train_size,
        val_size=0,
        seed=args.seed,
        seqlen=args.training_seqlen
    )
    
    # Optimizer with basic settings
    optimizer = torch.optim.AdamW(
        [model_with_prefix.prefix_embeddings], 
        lr=args.prefix_lr,
        weight_decay=1e-2,
        betas=(0.9, 0.999)
    )
    
    # Simple scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.prefix_lr,
        total_steps=args.prefix_epochs * len(train_loader),
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0,
        anneal_strategy='cos'
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    loss_fn = get_outlier_reduction_loss_fn(
        args.prefix_len, 
        outlier_threshold=args.outlier_threshold,
        target_layers=None
    )
    
    logger.info(f"[SoftPrefix] Starting training for {args.prefix_epochs} epochs...")
    
    model_with_prefix.train()
    best_loss = float('inf')
    best_prefix_embeddings = None
    
    for epoch in range(args.prefix_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                input_ids = batch[0].to(device)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    outputs = model_with_prefix(input_ids)
                    hidden_states_list = outputs.hidden_states[1:]
                    loss, _, _, _ = loss_fn(hidden_states_list, model_with_prefix.prefix_embeddings)
                
                if torch.isnan(loss):
                    logger.warning(f"[SoftPrefix] NaN loss at epoch {epoch}, batch {batch_idx}")
                    continue
                
                # Backward pass with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Update weights
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"[SoftPrefix] E{epoch}B{batch_idx}: loss={loss.item():.6f}, lr={scheduler.get_last_lr()[0]:.6f}")
                    
            except Exception as e:
                logger.warning(f"[SoftPrefix] Error in batch {batch_idx}: {str(e)}")
                optimizer.zero_grad()
                continue
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            logger.info(f"[SoftPrefix] Epoch {epoch}: loss={avg_loss:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_prefix_embeddings = model_with_prefix.prefix_embeddings.data.clone()
    
    if best_prefix_embeddings is not None:
        model_with_prefix.prefix_embeddings.data = best_prefix_embeddings
    
    logger.info("[SoftPrefix] Training completed")
    logger.info(f"[SoftPrefix] Best loss: {best_loss:.6f}")
    
    # Get closest tokens to trained embeddings
    with torch.no_grad():
        embed_layer = model.get_input_embeddings()
        trained_embeddings = model_with_prefix.prefix_embeddings.float()  # Ensure float32
        embed_weight = embed_layer.weight.float()  # Ensure float32
        
        normalized_prefix = F.normalize(trained_embeddings, p=2, dim=1)
        normalized_embeds = F.normalize(embed_weight, p=2, dim=1)
        
        similarities = torch.matmul(normalized_prefix, normalized_embeds.T)
        closest_token_ids = similarities.argmax(dim=1).tolist()
        
        logger.info(f"[SoftPrefix] Closest tokens: {closest_token_ids}")
        logger.info(f"[SoftPrefix] Token texts: {[tokenizer.decode([tid]) for tid in closest_token_ids]}")
    
    # Check if we need to collect activation statistics for static quantization
    include_static = (args.input_mode == "static" and args.input_bits < 16) or (args.kv_mode == "static" and (args.k_bits < 16 or args.v_bits < 16))
    activation_stat = None
    
    if include_static:
        logger.info("[SoftPrefix] Collecting activation statistics for static quantization...")
        cal_dataloader, _ = get_loaders(
            args.calib_dataset,
            tokenizer,
            train_size=args.train_size,
            val_size=0,
            seed=args.seed,
            seqlen=512,
        )
        # Get activation statistics using the learned prefix tokens
        activation_stat = get_act_stat(model, cal_dataloader, 'max', closest_token_ids, args.down_online_had)
        logger.info("[SoftPrefix] Activation statistics collected successfully")
    
    return closest_token_ids, activation_stat