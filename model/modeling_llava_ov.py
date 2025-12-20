# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForVision2Seq, LlavaOnevisionForConditionalGeneration, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ..decoderonly.modeling_decoderonly import RBLNDecoderOnlyOutput

import math
from torch import nn

import json
PROFILING_PATH = "profiling_log_info.jsonl"

def log_to_json(data: dict):
    with open(PROFILING_PATH, "a") as f:
        f.write(json.dumps(data) + "\n")


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig

# batch_size를 1 이상의 고정된 크기로 Loop 처리
class FixedLoopVisionTower:
    def __init__(self, vision_tower: RBLNModel, fixed_batch: int = 4) -> None:
        self.vision_tower = vision_tower
        self.fixed_batch = fixed_batch

    def forward(self, *args, **kwargs):
        pixel_values = args[0]

        batch_size = pixel_values.shape[0]
        outputs = []
        for i in range(0, batch_size, self.fixed_batch):
            chunk = pixel_values[i:i+self.fixed_batch]
            outputs.append(self.vision_tower.model[0](chunk))

        last_hidden_states = torch.cat([output[0] for output in outputs], dim=0)
        pooler_output = torch.cat([output[1] for output in outputs], dim=0)

        hidden_states = [output[2:] for output in outputs]
        hidden_states = tuple(
            torch.cat([hidden_states[c][i] for c in range(len(hidden_states))], dim=0)
            for i in range(len(hidden_states[0]))
        )

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_states,
            pooler_output=pooler_output,
            hidden_states=hidden_states,
        )
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        return repr(self.vision_tower)
        

class FixedLoopProjector:
    def __init__(self, multi_modal_projector, fixed_batch: int = 4) -> None:
        self.multi_modal_projector = multi_modal_projector
        self.fixed_batch = fixed_batch

    def forward(self, image_feature, *args, **kwargs):
        batch_size = image_feature.shape[0]
        outputs = []

        for i in range(0, batch_size, self.fixed_batch):
            chunk = image_feature[i:i+self.fixed_batch]
            outputs.append(self.multi_modal_projector(chunk))

        outputs = torch.cat(outputs, dim=0)
        return outputs
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def __repr__(self) -> str:
        return repr(self.multi_modal_projector)


# batch_size=1로 Loop 처리
class LoopVisionTower:
    def __init__(self, vision_tower: RBLNModel) -> None:
        self.vision_tower = vision_tower

    def forward(self, *args, **kwargs):
        # Loop instead of batch
        # shape of pixel_values : [batch, num_patches, num_channel, height, width]
        pixel_values = args[0]

        batch_size = pixel_values.shape[0]
        outputs = []
        for i in range(batch_size):
            outputs.append(self.vision_tower.model[0](pixel_values[i : i + 1]))
            

        last_hidden_states = [output[0] for output in outputs]
        pooler_output = [output[1] for output in outputs]

        # FIXME:: This can be optimized using out= API of rbln runtime.
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        pooler_output = torch.cat(pooler_output, dim=0)

        hidden_states = [output[2:] for output in outputs]  # batch x (hidden x 1)
    

        hidden_states = tuple(
            torch.cat(tuple((hidden_states[n][i] for n in range(batch_size))), dim=0)
            for i in range(len(hidden_states[0]))
        )  # hidden x (batch,)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_states,
            pooler_output=pooler_output,
            hidden_states=hidden_states,
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        return repr(self.vision_tower)


class LoopProjector:
    def __init__(self, multi_modal_projector) -> None:
        self.multi_modal_projector = multi_modal_projector

    def forward(self, *args, **kwargs):
        # Loop instead of batch
        image_feature = args[0]

        batch_size = image_feature.shape[0]
        outputs = []
        for i in range(batch_size):
            outputs.append(self.multi_modal_projector(image_feature[i : i + 1]))

        # FIXME:: This can be optimized using out= API of rbln runtime.
        outputs = torch.cat(outputs, dim=0)
        return outputs

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        return repr(self.multi_modal_projector)


class RBLNLlavaOnevisionForConditionalGeneration(RBLNModel):

    auto_model_class = AutoModelForVision2Seq
    _rbln_submodules = [
        {"name": "vision_tower"},
        {"name": "language_model"},
    ]

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(LlavaOnevisionForConditionalGeneration, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    def can_generate(self):
        return True

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "LlavaOnevisionForConditionalGeneration",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        # If you are unavoidably running on a CPU rather than an RBLN device,
        # store the torch tensor, weight, etc. in this function.
        save_dict = {}
        save_dict["image_newline"] = model.image_newline
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    def __post_init__(self, **kwargs):
        # final optim: batch_size=4
        self.vision_tower = FixedLoopVisionTower(self.rbln_submodules[0])
        self.multi_modal_projector = FixedLoopProjector(self.model[0])
        
        # batch=1
        # self.vision_tower = LoopVisionTower(self.rbln_submodules[0])
        # self.multi_modal_projector = LoopProjector(self.model[0])

        self.language_model = self.rbln_submodules[1]

        #self.vision_tower = self.rbln_submodules[0]
        #self.multi_modal_projector = self.model[0]
        
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.image_newline = artifacts["image_newline"]

        # Copied from the original class
        #self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        #self._padding_side = "left"  # set it to left by default, user can use setter to change padding_sides
        return super().__post_init__(**kwargs)

    def get_attn_impl(self) -> str:
        return self.rbln_config.language_model.attn_impl

    def get_kvcache_num_blocks(self) -> int:
        return self.rbln_config.language_model.kvcache_num_blocks

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    @classmethod
    def wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNModelConfig):
        return model.multi_modal_projector

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNModelConfig] = None,
    ) -> RBLNModelConfig:
        feature_size = model_config.vision_config.hidden_size

        # Siglip 형식 (SiglipVisionEmbedding)
        num_positions = (model_config.vision_config.image_size // model_config.vision_config.patch_size) ** 2
        if model_config.vision_feature_select_strategy == "default":
            selected_image_feature_dim = num_positions - 1
        else:
            selected_image_feature_dim = num_positions

        input_info = [
            ("image_features", [rbln_config.batch_size, selected_image_feature_dim, feature_size], "float32")
        ]
        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        pixel_values_videos=None,
        image_sizes_videos=None,
        attention_mask=None,
        generate_idx=None,
        **kwargs,
    ):
        # Prepare HF generation
        is_prefill_phase = generate_idx is None
        batch_size = input_ids.shape[0]

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            generate_idx=generate_idx,  # Not affect
            attention_mask=attention_mask,
            **kwargs,
        )

        if is_prefill_phase:
            model_inputs["generate_idx"] = torch.zeros((batch_size, 1), dtype=torch.int32)
            '''
            ## optim: patch pooling
            if pixel_values_videos is not None:
                pooled_frames = pixel_values_videos.shape[1]
                pooled_tokens = pooled_frames * 49 + 1
                video_token_idx = self.config.video_token_index
                mask = (input_ids == video_token_idx)
                if mask.any():
                    start_pos = torch.nonzero(mask, as_tuple=True)[1][0].item()
                    end_pos = torch.nonzero(mask, as_tuple=True)[1][-1].item()
                    new_input_ids = []
                    for i in range(batch_size):
                        row = input_ids[i]
                        new_row = torch.cat([
                            row[:start_pos],
                            row[start_pos:start_pos+pooled_tokens],
                            row[end_pos+1+mask.sum().item():]
                        ])
                        new_input_ids.append(new_row)
                    input_ids = torch.nn.utils.rnn.pad_sequence(
                        new_input_ids, batch_first=True, padding_value=0
                    ).to(input_ids.device)
                    new_attention_mask = []
                    for row in input_ids:
                        new_attention_mask.append(torch.ones_like(row))
                    attention_mask = torch.stack(new_attention_mask, dim=0).to(input_ids.device)
            '''

            model_inputs.update(
                {
                    "pixel_values": pixel_values,
                    "image_sizes": image_sizes,
                    "pixel_values_videos": pixel_values_videos,
                    "image_sizes_videos": image_sizes_videos,
                    #"input_ids": input_ids,     ## optim
                    #"attention_mask": attention_mask    ## optim
                }
            )

        model_inputs["attention_mask"] = attention_mask
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: RBLNDecoderOnlyOutput,
        model_kwargs: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        # update generate_idx
        model_kwargs["generate_idx"] = outputs.generate_idx

        return model_kwargs

    def text_embedding(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.Tensor:
        for_inputs_embeds_ids = input_ids.clone()
        for_inputs_embeds_ids[(input_ids == self.config.image_token_index)] = 0
        inputs_embeds = self.get_input_embeddings()(for_inputs_embeds_ids)

        return inputs_embeds

    # copied
    def pack_image_features(self, image_features, image_sizes, image_newline=None, vision_aspect_ratio="anyres_max_9"):
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                if height * width != base_image_feature.shape[0]:
                    raise ValueError("The number of patches is not consistent with the image size.")
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                max_num_patches = int(vision_aspect_ratio.strip("anyres_max_"))
                channels, curr_height, curr_width = image_feature.shape
                ratio = math.sqrt(curr_height * curr_width / (max_num_patches * height**2))
                if ratio > 1.1:
                    image_feature = image_feature[None]
                    image_feature = nn.functional.interpolate(
                        image_feature, [int(curr_height // ratio), int(curr_width // ratio)], mode="bilinear"
                    )[0]
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None]
                            .expand(*image_feature.shape[:-1], 1)
                            .to(image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens

    
    # copied
    def apply_pooling(self, image_features, height=None, width=None):   
        height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
        batch_frames, seq_len, dim = image_features.shape
        image_features = image_features.view(batch_frames, height, width, -1)
        image_features = image_features.permute(0, 3, 1, 2).contiguous()

        height, width = image_features.shape[2:]
        scaled_shape = [math.ceil(height / 2), math.ceil(width / 2)]
        image_features = nn.functional.interpolate(image_features, size=scaled_shape, mode="bilinear")

        image_features = image_features.permute(0, 2, 3, 1)
        image_features = image_features.view(batch_frames, -1, dim)
        return image_features

    def image_embedding(
        self,
        image_sizes: torch.Tensor,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int,
        vision_feature_select_strategy: str,
        vision_aspect_ratio: str,
    ):
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        vision_aspect_ratio = (
            vision_aspect_ratio if vision_aspect_ratio is not None else self.config.vision_aspect_ratio
        )

        # ! infer image_num_patches from image_sizes
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]
        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")
        image_features = self.vision_tower(pixel_values, output_hidden_states=True)


        selected_image_feature = image_features.hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = torch.split(image_features, image_num_patches, dim=0)

        # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
        image_features, feature_lens = self.pack_image_features(
                image_features,
                image_sizes,
                image_newline=self.image_newline,
                vision_aspect_ratio=vision_aspect_ratio,
        )

        return image_features, feature_lens
    
    def video_embedding(
    self,
    pixel_values_videos: torch.FloatTensor,
    vision_feature_layer: int = None,
    vision_feature_select_strategy: str = None,
    ):
        """
        pixel_values_videos: (batch_size, frames, 3, H, W)
        → return: (batch_size * frames + batch_size newline, feature_dim)
        """

        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        batch_size, frames, channels, height, width = pixel_values_videos.shape
        # (B, F, C, H, W) → (B*F, C, H, W)
        pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)
        video_features = self.vision_tower(pixel_values_videos, output_hidden_states=True)
        selected_video_feature = video_features.hidden_states[vision_feature_layer]

        if vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature
        video_features = self.multi_modal_projector(selected_video_feature)

        #video_features, h, w = self.apply_pooling(video_features)   ## h/w
        video_features = self.apply_pooling(video_features)
        # optim: patch pooling
        #video_features, h, w = self.apply_pooling(video_features, height=h, width=w)
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)
        image_newline = (
            self.image_newline[None, None, :].repeat(video_features.shape[0], 1, 1).to(video_features.device)
        )
        video_features = torch.cat((video_features, image_newline), dim=1)
        video_features = video_features.flatten(0, 1)

        return video_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_sizes_videos: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        vision_aspect_ratio: Optional[str] = None,
        cache_position: torch.Tensor = None,
        generate_idx: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
        **kwargs,
    ) -> Union[Tuple, RBLNDecoderOnlyOutput]:
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        vision_aspect_ratio = (
            vision_aspect_ratio if vision_aspect_ratio is not None else self.config.vision_aspect_ratio
        )

        if inputs_embeds is not None:
            raise NotImplementedError("Specifying inputs_embeds is not supported.")
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            # final optim: frame 분기
            #if not isinstance(self.vision_tower, LoopVisionTower):
            #    self.vision_tower = LoopVisionTower(self._vision_tower)
            #    self.multi_modal_projector = LoopProjector(self._multi_modal_projector)

            image_features, _ = self.image_embedding(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                vision_aspect_ratio=vision_aspect_ratio,
            )

            n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
            n_image_features = image_features.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            ## measure
            log_to_json({
                "step": "vision",
                "vision_tokens": int(n_image_tokens)
            })

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        
        if pixel_values_videos is not None:
            video_features = self.video_embedding(
                pixel_values_videos=pixel_values_videos,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

            n_video_tokens = (input_ids == self.config.video_token_index).sum().item()
            n_video_features = video_features.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            ## measure
            log_to_json({
                "step": "vision",
                "vision_tokens": int(n_video_tokens)
            })

            special_video_mask = (input_ids == self.config.video_token_index).unsqueeze(-1)
            special_video_mask = special_video_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)

        is_prefill_phase = not generate_idx.bool().all()

        if is_prefill_phase:
            logits = []
            batch_size = input_ids.shape[0]
            inputs_embeds = [inputs_embeds[i : i + 1, attention_mask[i].bool()] for i in range(batch_size)]
            for batch_idx in range(batch_size):
                ## measure
                log_to_json({
                    "step": "prefill",
                    "input_tokens": inputs_embeds[batch_idx].shape[-2]
                })
                generate_idx[batch_idx] = inputs_embeds[batch_idx].shape[-2]
                output = self.language_model.prefill_decoder(
                    inputs_embeds=inputs_embeds[batch_idx],
                    batch_idx=batch_idx,
                    cache_position=torch.arange(
                        0,
                        generate_idx[batch_idx].item(),
                        dtype=torch.int32,
                    ).unsqueeze(0),
                )

                logits.append(output.logits)
            logits = torch.cat(logits, dim=0)
        else:
            ## measure
            log_to_json({
                "step": "decode",
                "input_tokens": inputs_embeds[batch_idx].shape[-2]
            })

            output = self.language_model.decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
            )
            logits = output.logits
        return RBLNDecoderOnlyOutput(logits=logits, generate_idx=generate_idx)


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(image_size)} not valid, should be either list, tuple, np.ndarray or tensor"
            )
        image_size = image_size.tolist()

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size

def unpad_image(tensor, original_size):
    if not isinstance(original_size, (list, tuple)):
        if not isinstance(original_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(original_size)} not valid, should be either list, tuple, np.ndarray or tensor"
            )
        original_size = original_size.tolist()
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(round(original_height * scale_factor, 7))
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(round(original_width * scale_factor, 7))
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor

def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


# Almost copied from : https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/llava_next/modeling_llava_next.py
def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(f"image_size invalid type {type(image_size)} with value {image_size}")
        image_size = image_size.tolist()

    best_resolution = select_best_resolution(image_size, grid_pinpoints)
    height, width = best_resolution
    num_patches = 0
    # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    # add the base patch
    num_patches += 1
    return num_patches


