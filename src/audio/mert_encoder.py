import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from transformers.configuration_utils import PretrainedConfig

class MERTConfig(PretrainedConfig):
    r"""
    """
    model_type = "mert_model"

    def __init__(
        self,
        vocab_size=32,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        feat_proj_layer_norm=True,
        feat_proj_dropout=0.0,
        final_dropout=0.1,
        layerdrop=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        feat_extract_norm="group",
        feat_extract_activation="gelu",
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_bias=False,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        do_stable_layer_norm=False,
        apply_spec_augment=True,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_time_min_masks=2,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        mask_feature_min_masks=0,
        ctc_loss_reduction="sum",
        ctc_zero_infinity=False,
        use_weighted_layer_sum=False,
        classifier_proj_size=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        feature_extractor_cqt=False,
        feature_extractor_cqt_bins=336,
        deepnorm=False,
        attention_relax=-1.0,
        **kwargs
    ):
        super().__init__(**kwargs, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
        self.hidden_size = hidden_size
        self.feat_extract_norm = feat_extract_norm
        self.feat_extract_activation = feat_extract_activation
        self.conv_dim = list(conv_dim)
        self.conv_stride = list(conv_stride)
        self.conv_kernel = list(conv_kernel)
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_feat_extract_layers = len(self.conv_dim)
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.feat_proj_layer_norm = feat_proj_layer_norm
        self.feat_proj_dropout = feat_proj_dropout
        self.final_dropout = final_dropout
        self.layerdrop = layerdrop
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.do_stable_layer_norm = do_stable_layer_norm
        self.use_weighted_layer_sum = use_weighted_layer_sum
        self.classifier_proj_size = classifier_proj_size

        if (
            (len(self.conv_stride) != self.num_feat_extract_layers)
            or (len(self.conv_kernel) != self.num_feat_extract_layers)
            or (len(self.conv_dim) != self.num_feat_extract_layers)
        ):
            raise ValueError(
                "Configuration for convolutional layers is incorrect. It is required that `len(config.conv_dim)` =="
                " `len(config.conv_stride)` == `len(config.conv_kernel)`, but is `len(config.conv_dim) ="
                f" {len(self.conv_dim)}`, `len(config.conv_stride) = {len(self.conv_stride)}`,"
                f" `len(config.conv_kernel) = {len(self.conv_kernel)}`."
            )

        # fine-tuning config parameters for SpecAugment: https://arxiv.org/abs/1904.08779
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.mask_feature_min_masks = mask_feature_min_masks

        # ctc loss
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity

        # cqt feature extractor
        self.feature_extractor_cqt = feature_extractor_cqt
        self.feature_extractor_cqt_bins = feature_extractor_cqt_bins

        # deepnorm: up-scale weighted residual conection + down-scale initial value transformer encoder
        self.deepnorm = deepnorm

        self.attention_relax = attention_relax


class MERTEncoder(nn.Module):
    def __init__(self, model_name: str = "m-a-p/MERT-v1-95M", processor_name: str = "m-a-p/MERT-v1-95M", sampling_rate: int = None):
        super().__init__()
        # Initialize the processor and model
        config = MERTConfig.from_pretrained(model_name)
        config.conv_pos_batch_norm = False
        self.model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        # self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(processor_name, trust_remote_code=True)
        self.sampling_rate = self.processor.sampling_rate  # Default to processor's sampling rate
        if sampling_rate:
            self.sampling_rate = sampling_rate
    
    def load_audio(self, audio_file_name: str):
        # Load the audio file from the given path
        waveform, sampling_rate = torchaudio.load(audio_file_name)
        return waveform, sampling_rate
    
    def resample_audio(self, waveform: torch.Tensor, original_sampling_rate: int):
        # Resample the audio if necessary
        if original_sampling_rate != self.sampling_rate:
            # print(f'Resampling from {original_sampling_rate} to {self.sampling_rate}')
            resampler = T.Resample(original_sampling_rate, self.sampling_rate)
            waveform = resampler(waveform)
        return waveform
    
    def process_audio(self, audio_file_name: str):
        # Load and process the audio file
        waveform, sampling_rate = self.load_audio(audio_file_name)
        mono_waveform = waveform.mean(dim=0)
        # Resample if necessary
        waveform = self.resample_audio(mono_waveform, sampling_rate)
        
        # Process the waveform using the processor
        inputs = self.processor(waveform, sampling_rate=self.sampling_rate, return_tensors="pt")
        return inputs, waveform
    
    def get_hidden_states(self, audio_file_name: str):
        # Get the model's hidden states (features) for a given audio file
        inputs, waveform = self.process_audio(audio_file_name)
        # for key in inputs:
        #     inputs[key] = inputs[key].squeeze()
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract hidden states
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()

        # Time-reduced hidden states (mean pooling over time)
        time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
        
        return time_reduced_hidden_states[-1]