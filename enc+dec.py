import torch
from torch import nn
from typing import Optional, List, Tuple, Union
import math
from datasets import load_dataset

from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from transformers.models.bart.modeling_bart import (
    BartPreTrainedModel,
    BartConfig,
    BartLearnedPositionalEmbedding,
    BartEncoderLayer,
    BartDecoderLayer,
    BartEncoder,
    BartDecoder,
    BartModel,
    BartClassificationHead,
)
from transformers import BartConfig, BartTokenizer, Trainer, TrainingArguments,BartForSequenceClassification
import torch.utils.checkpoint
from sklearn.metrics import accuracy_score
import wandb

# Fourier Transform Implementations
class FourierFFTLayer(nn.Module):
    def forward(self, x):
        # Use FFT for the mixing operation
        return torch.fft.ifft(torch.fft.fft(x, dim=-1), dim=-1).real

class FourierMMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.register_buffer("fourier_matrix", self._make_fourier_matrix(self.d_model))
        
        # Proper scaling for better training stability
        with torch.no_grad():
            self.fourier_matrix = self.fourier_matrix / math.sqrt(self.d_model)

    def _make_fourier_matrix(self, d):
        i = torch.arange(d).unsqueeze(0)
        j = torch.arange(d).unsqueeze(1)
        omega = torch.exp(-2j * math.pi * i * j / d)
        return omega

    def forward(self, x):
        x_freq = torch.matmul(x, self.fourier_matrix.real)  # Only using real part for simplicity
        return x_freq

# FNet Layer for Encoder
class FNetEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        fourier_impl = getattr(config, "fourier_implementation", "fft")
        self.fft = FourierMMLayer(config) if fourier_impl == 'matmul' else FourierFFTLayer()
        
        self.mixing_layer_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.output_dense = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.output_layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        if isinstance(config.activation_function, str):
            act = config.activation_function.lower()
            if act == "relu":
                self.activation = nn.ReLU()
            elif act in {"silu", "swish"}:
                self.activation = nn.SiLU()
            else:
                self.activation = nn.GELU()
        else:
            self.activation = config.activation_function

    def forward(self, hidden_states, attention_mask=None):
        # Self-mixing with Fourier Transform - with proper residual connection
        residual = hidden_states
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask from [batch_size, seq_len] to [batch_size, seq_len, 1]
            mask_expanded = attention_mask.unsqueeze(-1).float()
            hidden_states = hidden_states * mask_expanded
            
        fft_output = self.fft(hidden_states)
        hidden_states = self.mixing_layer_norm(fft_output + residual)
        
        # Feed forward network with residual connection
        residual = hidden_states
        intermediate_output = self.feed_forward(hidden_states)
        activated_output = self.activation(intermediate_output)
        projected_output = self.output_dense(activated_output)
        dropped_output = self.dropout(projected_output)
        hidden_states = self.output_layer_norm(dropped_output + residual)
        
        return hidden_states

# FNet Layer for Decoder
class FNetDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        fourier_impl = getattr(config, "fourier_implementation", "fft")
        self.fft = FourierMMLayer(config) if fourier_impl == 'matmul' else FourierFFTLayer()

        # Layer for incorporating encoder information without attention
        self.encoder_projection = nn.Linear(config.d_model, config.d_model)
        self.encoder_mixing_layer_norm = nn.LayerNorm(config.d_model)
        
        self.mixing_layer_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.output_dense = nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.output_layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        if isinstance(config.activation_function, str):
            act = config.activation_function.lower()
            if act == "relu":
                self.activation = nn.ReLU()
            elif act in {"silu", "swish"}:
                self.activation = nn.SiLU()
            else:
                self.activation = nn.GELU()
        else:
            self.activation = config.activation_function

    def forward(self, hidden_states, encoder_hidden_states=None, encoder_attention_mask=None, attention_mask=None):
        # Self-mixing with Fourier Transform - with proper residual connection
        residual = hidden_states
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask from [batch_size, seq_len] to [batch_size, seq_len, 1]
            mask_expanded = attention_mask.unsqueeze(-1).float()
            hidden_states = hidden_states * mask_expanded
            
        fft_output = self.fft(hidden_states)
        normed_fft_output = self.mixing_layer_norm(fft_output + residual)
        
        # Incorporate encoder information without attention
        if encoder_hidden_states is not None:
            residual = normed_fft_output
            
            # Use encoder attention mask to properly mask padding tokens
            if encoder_attention_mask is not None:
                # Convert mask from [batch_size, seq_len] to [batch_size, seq_len, 1]
                mask_expanded = encoder_attention_mask.unsqueeze(-1).float()
                # Apply mask to encoder states
                masked_encoder = encoder_hidden_states * mask_expanded
                # Calculate sum and count of non-masked elements
                encoder_sum = torch.sum(masked_encoder, dim=1, keepdim=True)
                mask_sum = torch.sum(mask_expanded, dim=1, keepdim=True).clamp(min=1e-9)
                # Average considering only non-masked elements
                encoder_avg = encoder_sum / mask_sum
            else:
                # Simple average if no mask is provided
                encoder_avg = torch.mean(encoder_hidden_states, dim=1, keepdim=True)
                
            # Expand to match sequence length of decoder
            seq_len = hidden_states.size(1)
            encoder_proj = self.encoder_projection(encoder_avg)
            encoder_proj = encoder_proj.expand(-1, seq_len, -1)
            
            # Add encoder representation to current hidden state (instead of concatenating)
            hidden_states = normed_fft_output + encoder_proj
            hidden_states = self.encoder_mixing_layer_norm(hidden_states + residual)
        else:
            hidden_states = normed_fft_output

        # Feed forward network with residual connection
        residual = hidden_states
        intermediate_output = self.feed_forward(hidden_states)
        activated_output = self.activation(intermediate_output)
        projected_output = self.output_dense(activated_output)
        dropped_output = self.dropout(projected_output)
        hidden_states = self.output_layer_norm(dropped_output + residual)
        
        return hidden_states

# Complete FNet Encoder 
class FNetEncoder(BartPreTrainedModel):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        
        # Embedding setup with proper scaling
        embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.embed_scale = embed_scale

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([FNetEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        
        # Final layer norm for better stability
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        
        self.gradient_checkpointing = False
        
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Check input configurations
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Create embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # Add positional embeddings
        embed_pos = self.embed_positions(input_ids)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # Process through layers
        all_hidden_states = () if output_hidden_states else None

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Apply layer drop if enabled
            dropout_probability = torch.rand([])
            if self.training and dropout_probability < self.layerdrop:
                continue

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                )
            else:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask
                )

        # Apply final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
            
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=None,  # No attention outputs in FNet
        )

# Complete FNet Decoder
class FNetDecoder(BartPreTrainedModel):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.layerdrop = config.decoder_layerdrop
        
        # Embedding setup with proper scaling
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_scale = embed_scale
        
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(config.max_position_embeddings, config.d_model)
        self.layers = nn.ModuleList([FNetDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        
        # Add final layer norm for better stability
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # Embedding step
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # Add positional embeddings
        positions = self.embed_positions(input_ids)
        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        all_hidden_states = () if output_hidden_states else None

        # Process through layers
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            # Apply layer drop if enabled
            dropout_probability = torch.rand([])
            if self.training and dropout_probability < self.layerdrop:
                continue

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    attention_mask,
                )
            else:
                # Pass encoder outputs to each layer
                layer_outputs = decoder_layer(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    attention_mask=attention_mask
                )
                
            hidden_states = layer_outputs

        # Apply final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, None, None] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            past_key_values=None,  # FNet doesn't use this
            attentions=None,       # No self-attention here
            cross_attentions=None  # Optional, also not used
        )

# Full BART model with FNet replacing attention
class FNetBartModel(BartPreTrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = FNetEncoder(config, self.shared)
        self.decoder = FNetDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError("`config.decoder_start_token_id` has to be defined.")

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("`config.pad_token_id` has to be defined.")
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = self._shift_right(input_ids)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return dict(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

# BART model with classification head
class FNetBartForSequenceClassification(BartPreTrainedModel):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = FNetBartModel(config)
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state

        # Extract features from the first token for classification (like BERT's [CLS] token)
        eos_mask = input_ids.eq(self.config.eos_token_id) if input_ids is not None else \
            decoder_input_ids.eq(self.config.eos_token_id)
        
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
            
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return dict(
            loss=loss,
            logits=logits,
            past_key_values=outputs.get("past_key_values", None),
            decoder_hidden_states=outputs.get("decoder_hidden_states", None),
            decoder_attentions=outputs.get("decoder_attentions", None),
            cross_attentions=outputs.get("cross_attentions", None),
            encoder_last_hidden_state=outputs.get("encoder_last_hidden_state", None),
            encoder_hidden_states=outputs.get("encoder_hidden_states", None),
            encoder_attentions=outputs.get("encoder_attentions", None),
        )

def init_fnet_properly(model):
    """Initialize FNet layers with appropriate scaling"""
    for name, module in model.named_modules():
        # Linear layers
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        # LayerNorm
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# Main training code
def main():
    # Initialize wandb
    wandb.init(project="sst2-bart-finetuning", name="bart-enc+dec-fnet-final")
    
    # Load tokenizer and config
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    config = BartConfig.from_pretrained("facebook/bart-base", num_labels=2)
    
    # Create FNet BART model
    model = BartForSequenceClassification.from_pretrained("facebook/bart-base", num_labels=2)
    shared_embeddings = model.get_input_embeddings()
    fnet_encoder = FNetEncoder(config=config, embed_tokens=shared_embeddings)
    model.model.encoder = fnet_encoder
    model.model.decoder = FNetDecoder(config=config, embed_tokens=model.get_input_embeddings())
    # Properly initialize the FNet layers
    init_fnet_properly(model)
    
    # Load SST-2 dataset
    dataset = load_dataset("glue", "sst2")
    
    # Tokenization function
    def preprocess_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    # Tokenize the dataset
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Split datasets
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./bart_fnet_full_results",
        eval_strategy="epoch",
        eval_steps=200,
        learning_rate=5e-5,  # Slightly higher learning rate for full training
        lr_scheduler_type="cosine",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs_bart_fnet_full",
        report_to="wandb",
        logging_steps=50,
        save_steps=200,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        warmup_ratio=0.1,
        fp16=True,
    )

    # Accuracy metric
    def compute_metrics(p):
        if isinstance(p.predictions, tuple):
            preds = p.predictions[0]
        else:
            preds = p.predictions

        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        preds = preds.argmax(axis=1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}

    # Training
    print("Training full FNet-BART model")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # Save the final model
    trainer.save_model("./sst2_full_fnet_bart_final")
    
    # Evaluate on validation set
    results = trainer.evaluate()
    print("Final evaluation results:", results)
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()