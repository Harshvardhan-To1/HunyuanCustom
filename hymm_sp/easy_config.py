from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import argparse
import re
import collections.abc
from hymm_sp.constants import *


def as_tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    if x is None or isinstance(x, (int, float, str)):
        return (x,)
    else:
        raise ValueError(f"Unknown type {type(x)}")


@dataclass
class HYMMConfig:
    # Network arguments
    model: str = "HYVideo-T/2"
    latent_channels: Optional[int] = None
    rope_theta: int = 256
    
    # Extra Models (VAE, Text Encoder, Tokenizer)
    vae: str = "884-16c-hy0801"
    vae_precision: str = "fp16"
    vae_tiling: bool = True
    text_encoder: str = "llava-llama-3-8b"
    text_encoder_precision: str = "fp16"
    text_states_dim: int = 4096
    text_len: int = 256
    tokenizer: str = "llava-llama-3-8b"
    text_encoder_infer_mode: str = "encoder"
    prompt_template_video: str = 'li-dit-encode-video'
    hidden_state_skip_layer: int = 2
    apply_final_norm: bool = False
    text_encoder_2: str = 'clipL'
    text_encoder_precision_2: str = "fp16"
    text_states_dim_2: int = 768
    tokenizer_2: str = 'clipL'
    text_len_2: int = 77
    use_attention_mask: bool = True
    text_projection: str = "single_refiner"
    
    # Denoise schedule arguments
    flow_shift_eval_video: Optional[float] = 13.0
    flow_reverse: bool = True
    flow_solver: str = "euler"
    use_linear_quadratic_schedule: bool = False
    linear_schedule_end: int = 25
    
    # Evaluation arguments
    precision: str = "bf16"
    reproduce: bool = False
    ckpt: Optional[str] = None
    load_key: str = "module"
    cpu_offload: bool = False
    use_fp8: bool = False
    video_size: Union[int, List[int]] = field(default_factory=lambda: [512])
    sample_n_frames: int = 1
    infer_steps: int = 100
    val_disable_autocast: bool = False
    num_images: int = 1
    seed: int = 1024
    save_path_suffix: str = ""
    pos_prompt: str = ''
    neg_prompt: str = ''
    add_pos_prompt: str = ''
    add_neg_prompt: str = ''
    pad_face_size: float = 0.7
    image_path: str = ""
    save_path: Optional[str] = None
    input: Optional[str] = None
    item_name: Optional[str] = None
    cfg_scale: float = 7.5
    ip_cfg_scale: float = 0
    use_deepcache: int = 1
    
    def __post_init__(self):
        # Handle video_size as either single int or list
        if isinstance(self.video_size, int):
            self.video_size = [self.video_size, self.video_size]
        
        # Perform sanity checks
        self._sanity_check()
    
    def _sanity_check(self):
        # VAE channels check
        vae_pattern = r"\d{2,3}-\d{1,2}c-\w+"
        if not re.match(vae_pattern, self.vae):
            raise ValueError(
                f"Invalid VAE model: {self.vae}. Must be in the format of '{vae_pattern}'."
            )
        vae_channels = int(self.vae.split("-")[1][:-1])
        if self.latent_channels is None:
            self.latent_channels = vae_channels
        if vae_channels != self.latent_channels:
            raise ValueError(
                f"Latent channels ({self.latent_channels}) must match the VAE channels ({vae_channels})."
            )
    
    @classmethod
    def from_args(cls, args=None, namespace=None):
        """
        Create a HYMMConfig instance from command line arguments.
        """
        parser = argparse.ArgumentParser(description="Hunyuan Multimodal training/inference script")
        parser = cls._add_all_arguments(parser)
        
        if args is not None or namespace is not None:
            parsed_args = parser.parse_args(args=args, namespace=namespace)
            # Convert namespace to dictionary
            args_dict = vars(parsed_args)
            return cls(**args_dict)
        else:
            # If no args provided, return the default config
            return cls()
    
    @classmethod
    def _add_all_arguments(cls, parser):
        """Add all arguments to the parser"""
        parser = cls._add_network_args(parser)
        parser = cls._add_extra_models_args(parser)
        parser = cls._add_denoise_schedule_args(parser)
        parser = cls._add_evaluation_args(parser)
        return parser
    
    @staticmethod
    def _add_network_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Network")
        group.add_argument("--model", type=str, default="HYVideo-T/2",
                        help="Model architecture to use. It it also used to determine the experiment directory.")
        group.add_argument("--latent-channels", type=str, default=None,
                        help="Number of latent channels of DiT. If None, it will be determined by `vae`. If provided, "
                                "it still needs to match the latent channels of the VAE model.")
        group.add_argument("--rope-theta", type=int, default=256, help="Theta used in RoPE.")
        return parser

    @staticmethod
    def _add_extra_models_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Extra Models (VAE, Text Encoder, Tokenizer)")

        # VAE
        group.add_argument("--vae", type=str, default="884-16c-hy0801",  help="Name of the VAE model.")
        group.add_argument("--vae-precision", type=str, default="fp16", 
                        help="Precision mode for the VAE model.")
        group.add_argument("--vae-tiling", action="store_true", default=True, help="Enable tiling for the VAE model.")
        group.add_argument("--text-encoder", type=str, default="llava-llama-3-8b", choices=list(TEXT_ENCODER_PATH),
                        help="Name of the text encoder model.")
        group.add_argument("--text-encoder-precision", type=str, default="fp16", choices=PRECISIONS,
                        help="Precision mode for the text encoder model.")
        group.add_argument("--text-states-dim", type=int, default=4096, help="Dimension of the text encoder hidden states.")
        group.add_argument("--text-len", type=int, default=256, help="Maximum length of the text input.")
        group.add_argument("--tokenizer", type=str, default="llava-llama-3-8b", choices=list(TOKENIZER_PATH),
                        help="Name of the tokenizer model.")
        group.add_argument("--text-encoder-infer-mode", type=str, default="encoder", choices=["encoder", "decoder"],
                        help="Inference mode for the text encoder model. It should match the text encoder type. T5 and "
                                "CLIP can only work in 'encoder' mode, while Llava/GLM can work in both modes.")
        group.add_argument("--prompt-template-video", type=str, default='li-dit-encode-video', choices=PROMPT_TEMPLATE,
                        help="Video prompt template for the decoder-only text encoder model.")
        group.add_argument("--hidden-state-skip-layer", type=int, default=2,
                        help="Skip layer for hidden states.")
        group.add_argument("--apply-final-norm", action="store_true",
                        help="Apply final normalization to the used text encoder hidden states.")

        # - CLIP
        group.add_argument("--text-encoder-2", type=str, default='clipL', choices=list(TEXT_ENCODER_PATH),
                        help="Name of the second text encoder model.")
        group.add_argument("--text-encoder-precision-2", type=str, default="fp16", choices=PRECISIONS,
                        help="Precision mode for the second text encoder model.")
        group.add_argument("--text-states-dim-2", type=int, default=768,
                        help="Dimension of the second text encoder hidden states.")
        group.add_argument("--tokenizer-2", type=str, default='clipL', choices=list(TOKENIZER_PATH),
                        help="Name of the second tokenizer model.")
        group.add_argument("--text-len-2", type=int, default=77, help="Maximum length of the second text input.")
        group.set_defaults(use_attention_mask=True)
        group.add_argument("--text-projection", type=str, default="single_refiner", choices=TEXT_PROJECTION,
                        help="A projection layer for bridging the text encoder hidden states and the diffusion model "
                                "conditions.")
        return parser

    @staticmethod
    def _add_denoise_schedule_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Denoise schedule")
        group.add_argument("--flow-shift-eval-video", type=float, default=None, help="Shift factor for flow matching schedulers when using video data.")
        group.add_argument("--flow-reverse", action="store_true", default=True, help="If reverse, learning/sampling from t=1 -> t=0.")
        group.add_argument("--flow-solver", type=str, default="euler", help="Solver for flow matching.")
        group.add_argument("--use-linear-quadratic-schedule", action="store_true", help="Use linear quadratic schedule for flow matching."
                                                        "Follow MovieGen (https://ai.meta.com/static-resource/movie-gen-research-paper)")
        group.add_argument("--linear-schedule-end", type=int, default=25, help="End step for linear quadratic schedule for flow matching.")
        return parser

    @staticmethod
    def _add_evaluation_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Validation Loss Evaluation")
        parser.add_argument("--precision", type=str, default="bf16", choices=PRECISIONS,
                        help="Precision mode. Options: fp32, fp16, bf16. Applied to the backbone model and optimizer.")
        parser.add_argument("--reproduce", action="store_true",
                        help="Enable reproducibility by setting random seeds and deterministic algorithms.")
        parser.add_argument("--ckpt", type=str, help="Path to the checkpoint to evaluate.")
        parser.add_argument("--load-key", type=str, default="module", choices=["module", "ema"],
                        help="Key to load the model states. 'module' for the main model, 'ema' for the EMA model.")
        parser.add_argument("--cpu-offload", action="store_true", help="Use CPU offload for the model load.")
        group.add_argument( "--use-fp8", action="store_true", help="Enable use fp8 for inference acceleration.")
        group.add_argument("--video-size", type=int, nargs='+', default=512,
                            help="Video size for training. If a single value is provided, it will be used for both width "
                                "and height. If two values are provided, they will be used for width and height "
                                "respectively.")
        group.add_argument("--sample-n-frames", type=int, default=1,
                        help="How many frames to sample from a video. if using 3d vae, the number should be 4n+1")
        group.add_argument("--infer-steps", type=int, default=100, help="Number of denoising steps for inference.")
        group.add_argument("--val-disable-autocast", action="store_true",
                        help="Disable autocast for denoising loop and vae decoding in pipeline sampling.")
        group.add_argument("--num-images", type=int, default=1, help="Number of images to generate for each prompt.")
        group.add_argument("--seed", type=int, default=1024, help="Seed for evaluation.")
        group.add_argument("--save-path-suffix", type=str, default="", help="Suffix for the directory of saved samples.")
        group.add_argument("--pos-prompt", type=str, default='', help="Prompt for sampling during evaluation.")
        group.add_argument("--neg-prompt", type=str, default='', help="Negative prompt for sampling during evaluation.")
        group.add_argument("--add-pos-prompt", type=str, default='', help="Addition prompt for sampling during evaluation.")
        group.add_argument("--add-neg-prompt", type=str, default='', help="Addition negative prompt for sampling during evaluation.")
        group.add_argument("--pad-face-size", type=float, default=0.7, help="Pad bbox for face align.")
        group.add_argument("--image-path", type=str, default="",  help="")
        group.add_argument("--save-path", type=str, default=None, help="Path to save the generated samples.")
        group.add_argument("--input", type=str, default=None, help="test data.")
        group.add_argument("--item-name", type=str, default=None, help="")
        group.add_argument("--cfg-scale", type=float, default=7.5, help="Classifier free guidance scale.")
        group.add_argument("--ip-cfg-scale", type=float, default=0, help="Classifier free guidance scale.")
        group.add_argument("--use-deepcache", type=int, default=1)
        return parser
    
    def to_dict(self):
        """Convert the config to a dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def update(self, **kwargs):
        """Update config with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        # Run sanity checks after update
        self._sanity_check()
        return self


if __name__ == "__main__":
    models_base = "./models"
    config = HYMMConfig(
        input="./assets/images/seg_woman_01.png",
        pos_prompt="Realistic, High-quality. A woman is drinking coffee at a café.",
        neg_prompt="Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.",
        ckpt=f"{models_base}/hunyuancustom_720P/mp_rank_00_model_states.pt",
        video_size=(512, 896),
        seed=1024,
        sample_n_frames=129,
        infer_steps=30,
        flow_shift_eval_video=13.0,
        save_path='./results/new_video_540p'
    )

    print(config.video_size, config.infer_steps)
