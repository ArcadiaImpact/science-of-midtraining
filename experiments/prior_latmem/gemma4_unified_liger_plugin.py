"""Axolotl plugin for memory-bounded Gemma 4 Unified causal-LM loss.

Transformers' unified Gemma 4 wrapper materializes ``[batch, sequence,
vocabulary]`` logits before calling its loss function.  A retained 8k-token
training example therefore exceeds even a 94GB H100.  Liger's fused linear
cross-entropy computes the same masked causal objective directly from hidden
states and the LM-head weight, without constructing that tensor.

This patch is training-only.  Evaluation and generation retain Transformers'
normal logits path, while vLLM generation runs in a separate environment and
never imports this module.
"""

from __future__ import annotations

from axolotl.integrations.base import BasePlugin


class Gemma4UnifiedLigerPlugin(BasePlugin):
    """Install fused linear CE on ``Gemma4UnifiedForConditionalGeneration``."""

    def pre_model_load(self, cfg) -> None:
        import torch
        from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
        from transformers.models.gemma4_unified import modeling_gemma4_unified

        output_type = modeling_gemma4_unified.Gemma4UnifiedCausalLMOutputWithPast

        def fused_forward(
            self,
            input_ids: torch.LongTensor | None = None,
            pixel_values: torch.FloatTensor | None = None,
            pixel_values_videos: torch.FloatTensor | None = None,
            input_features: torch.FloatTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            input_features_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            image_position_ids: torch.LongTensor | None = None,
            video_position_ids: torch.LongTensor | None = None,
            past_key_values=None,
            mm_token_type_ids: torch.LongTensor | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            labels: torch.LongTensor | None = None,
            use_cache: bool | None = None,
            logits_to_keep: int | torch.Tensor = 0,
            **kwargs,
        ):
            # PEFT forwards this as a kwarg even though the unified wrapper
            # always requests a structured inner-model result explicitly.
            kwargs.pop("return_dict", None)
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                input_features=input_features,
                attention_mask=attention_mask,
                input_features_mask=input_features_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                image_position_ids=image_position_ids,
                video_position_ids=video_position_ids,
                return_dict=True,
                **kwargs,
            )
            hidden_states = outputs.last_hidden_state
            text_config = self.config.get_text_config()
            final_softcap = text_config.final_logit_softcapping

            if self.training and labels is not None:
                logits = None
                loss = LigerForCausalLMLoss(
                    hidden_states=hidden_states,
                    lm_head_weight=self.lm_head.weight,
                    labels=labels,
                    hidden_size=hidden_states.shape[-1],
                    final_logit_softcapping=final_softcap,
                    **kwargs,
                )
            else:
                slice_indices = (
                    slice(-logits_to_keep, None)
                    if isinstance(logits_to_keep, int)
                    else logits_to_keep
                )
                logits = self.lm_head(hidden_states[:, slice_indices, :])
                if final_softcap is not None:
                    logits = torch.tanh(logits / final_softcap) * final_softcap
                loss = None
                if labels is not None:
                    loss = self.loss_function(
                        logits,
                        labels,
                        text_config.vocab_size,
                        **kwargs,
                    )

            return output_type(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                image_hidden_states=outputs.image_hidden_states,
                audio_hidden_states=outputs.audio_hidden_states,
                shared_kv_states=outputs.shared_kv_states,
            )

        modeling_gemma4_unified.Gemma4UnifiedForConditionalGeneration.forward = (
            fused_forward
        )


__all__ = ["Gemma4UnifiedLigerPlugin"]
