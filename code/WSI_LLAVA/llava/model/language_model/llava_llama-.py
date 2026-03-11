#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import math

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.kg_embeddings = None
        self.kg_projector = None
        self.kg_global_projector = None
        
    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            # print("**********1111inputs_embeds1111**********",inputs_embeds,shape)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
    def _get_kg_bank(self):
        model = self.get_model()
        if not getattr(model.config, "use_kg", False):
            return None

        kg = getattr(model, "kg_embeddings", None)
        if kg is None:
            return None

        # 这里不做 .to()，不做 projector，完全返回原始 KG
        return kg

    def _pool_query(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        """
        seq_embeds: [T, D]
        return q: [D]
        """
        pool = getattr(self.config, "kg_query_pool", "mean")
        if pool == "last":
            return seq_embeds[-1]
        # default mean
        return seq_embeds.mean(dim=0)

    @torch.no_grad()
    def _retrieve_kg_tokens(self, seq_embeds: torch.Tensor, max_k: int | None = None):
        """
        return:
        kg_tokens: [4, D]   (static top-2 + dynamic top-2)
        topk_idx:  [4]
        """
        kg = self._get_kg_bank()
        if kg is None:
            return None, None

        device = next(self.parameters()).device
        kg = kg.to(device)
        kg_f = kg.float()

        static_k = XXXX

        q = self._pool_query(seq_embeds).float()


        static_tokens = kg[static_idx]   # 原 dtype

        if getattr(self.config, "kg_weighted", True):
            w = torch.softmax(static_scores_k, dim=-1).to(static_tokens.dtype)
            static_tokens = static_tokens * w.unsqueeze(-1)

        dynamic_k = XXXX

        ctx = self._pool_query(seq_embeds).float()
        ctx = ctx / (ctx.norm(dim=-1, keepdim=True) + 1e-6)

        kg_normed = kg_f / (kg_f.norm(dim=-1, keepdim=True) + 1e-6)


        return static_tokens, topk_idx

        
###############################################################################                
    def _project_kg_global(self, kg: torch.Tensor) -> torch.Tensor:
        """
        Global KG projector（第二套 MLP）
        注意：真正的 projector 权重存在 self.get_model().kg_global_projector 中
        """
        model = self.get_model()
        return model.kg_global_projector(kg)


    # def _retrieve_global_kg_tokens(self, max_k=None):
    #     """
    #     Global importance KG retrieval (Top-K)
    #     """
    #     model = self.get_model()
    #     kg_raw = model.kg_embeddings
    #     if kg_raw is None:
    #         return None, None

    #     top_k = max_k or getattr(model.config, "kg_global_top_k", 2)

    #     device = next(model.parameters()).device
    #     kg_raw = kg_raw.to(device)

    #     # ================= 新增：图文 global context =================
    #     # 约定：在外部已缓存当前图文 embedding
    #     mm_embeds = getattr(self, "_last_mm_embeds", None)

    #     if mm_embeds is not None:
    #         # pooling 成一个全局向量
    #         ctx = self._pool_query(mm_embeds).float()   # [D]
    #         ctx = ctx.to(device)

    #         # broadcast 到每个 KG
    #         ctx = ctx.unsqueeze(0).expand_as(kg_raw)    # [N, D]

    #         # KG + 图文 context 一起送入 projector
    #         kg_input = kg_raw + ctx
    #     else:
    #         kg_input = kg_raw
    #     # ============================================================
    #     # print("kg_input dtype:", kg_input.dtype)
    #     # print("kg_global_projector weight dtype:",
    #     #     model.kg_global_projector[0].weight.dtype)
    #     # 1) global projection（现在“看见”了图文）
    #     # proj = model.kg_global_projector(kg_input)  # [N, hidden]
    #     proj = model.kg_global_projector(
    #         kg_input.to(model.kg_global_projector[0].weight.dtype)
    #     )
    #     # 2) importance score：L2 norm（不算相似度）
    #     proj = proj.float()  
    #     scores = proj.norm(dim=-1)  # [N]

    #     # 3) Top-K
    #     topk_scores, topk_idx = torch.topk(scores, k=top_k, largest=True)

    #     # 4) softmax weighting
    #     # w = torch.softmax(topk_scores, dim=-1).unsqueeze(-1)  # [K,1]

    #     # tokens = proj[topk_idx] * w
    #     w = torch.softmax(topk_scores.float(), dim=-1).unsqueeze(-1)
    #     tokens = proj[topk_idx] * w.to(proj.dtype)
    #     return tokens, topk_idx

###############################################################################                
AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
