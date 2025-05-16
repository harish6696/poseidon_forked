"""
Our version of the Huggingface Trainer class.
It adds learning_rate_time_embedding, learning_rate_embedding_recovery as 
additional learning rates and groups parameters for the optimizer.
It also allows for autoregressive rollouts by using 
trainer.set_ar_steps(AR_STEPS) where AR_STEPS is either a an integer for a 
homogeneous rollout of AR_STEPS steps or a list of integers for a heterogeneous
rollout where each element is the timestep.
If, additionally, output_all_steps is also set, the predict function will
output all intermediate steps as well.

We sublass a Huggingface Trainer to allow for autoregressive rollouts and multiple parameter groups in the optimizer.
It is specifically subclassed for our purpose.

A lot of code is copied over because only slight changes have been made.

The original code of Huggingface Transformers is distributed under the Apache 2.0 license. See below:

Copyright 2018- The Hugging Face team. All rights reserved.

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

"""

import torch
from torch import nn
from typing import List, Optional, Dict, Tuple, Union, Any
from transformers.trainer import *
from transformers import Trainer as Trainer_
from transformers import TrainingArguments as TrainingArguments_
from scOT.model import LayerNorm, ConditionalLayerNorm
from dataclasses import dataclass, field


@dataclass
class TrainingArguments(TrainingArguments_):
    learning_rate_embedding_recovery: Optional[float] = field(
        default=None,
        metadata={
            "help": "The initial learning rate for the embedding/recovery. When not provided, falls back to `learning_rate`."
        },
    )

    learning_rate_time_embedding: Optional[float] = field(
        default=None,
        metadata={
            "help": "The initial learning rate for the time embedding. When not provided, falls back to `learning_rate`. Only used when embedding and recovery are also fine-tuned with different lr."
        },
    )

    def set_training(  ##overrides the one in the base class from transformers library
        self,
        *args,
        learning_rate_embedding_recovery: Optional[float] = None,
        learning_rate_time_embedding: Optional[float] = None,
        **kwargs,
    ):
        self = super().set_training(*args, **kwargs)
        self.learning_rate_embedding_recovery = learning_rate_embedding_recovery
        self.learning_rate_time_embedding = learning_rate_time_embedding
        return self

    def set_optimizer(  ##overrides the one in the base class from transformers library
        self,
        *args,
        learning_rate_embedding_recovery: Optional[float] = None,
        learning_rate_time_embedding: Optional[float] = None,
        **kwargs,
    ):
        self = super().set_optimizer(*args, **kwargs)
        self.learning_rate_embedding_recovery = learning_rate_embedding_recovery
        self.learning_rate_time_embedding = learning_rate_time_embedding
        return self


class Trainer(Trainer_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ar_steps = None
        self.output_all_steps = False

    def get_decay_parameter_names(self, model) -> List[str]: #used inside create_optimizer method below (not overriden)
        ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm, LayerNorm, ConditionalLayerNorm]
        decay_parameters = get_parameter_names(model, forbidden_layer_types=ALL_LAYERNORM_LAYERS) #inside this function, we go layer by layer and add the parameters excluding the Layernorm layers
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters

    def get_conditional_norm_params(self, model): #used inside create_optimizer method below (not ovverriden)
        params = []
        for name, module in model.named_modules():
            if isinstance(module, ConditionalLayerNorm):
                for param_name, _ in module.named_parameters():
                    params.append(f"{name}.{param_name}")
        return params

    def create_optimizer(self):   #overrides the one in the base class from transformers library
        """This is the same as in the standard trainer, except param groups"""
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model #opt_model is the ScOT model object 
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)
            if self.args.learning_rate_embedding_recovery is not None: ##only used for finetuning..
                if self.args.learning_rate_time_embedding is not None:
                    time_embedding_params = self.get_conditional_norm_params(self.model)
                    params = {
                        "standard": [],
                        "no_weight_decay": [],
                        "embeddings": [],
                        "time_embedding": [],
                    }
                    for n, p in opt_model.named_parameters():
                        if (
                            "embeddings" in n or "patch_recovery" in n
                        ) and p.requires_grad:
                            params["embeddings"].append(p)
                        elif n in decay_parameters and p.requires_grad:
                            params["standard"].append(p)
                        elif p.requires_grad:
                            if n in time_embedding_params:
                                params["time_embedding"].append(p)
                            else:
                                params["no_weight_decay"].append(p)
                    optimizer_grouped_parameters = [
                        {
                            "params": params["standard"],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": params["no_weight_decay"],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": params["embeddings"],
                            "lr": self.args.learning_rate_embedding_recovery,
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": params["time_embedding"],
                            "lr": self.args.learning_rate_time_embedding,
                            "weight_decay": 0.0,
                        },
                    ]
                else:
                    params = {"standard": [], "no_weight_decay": [], "embeddings": []}
                    for n, p in opt_model.named_parameters():
                        if (
                            "embeddings" in n or "patch_recovery" in n
                        ) and p.requires_grad:
                            params["embeddings"].append(p)
                        elif n in decay_parameters and p.requires_grad:
                            params["standard"].append(p)
                        elif p.requires_grad:
                            params["no_weight_decay"].append(p)
                    optimizer_grouped_parameters = [
                        {
                            "params": params["standard"],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": params["no_weight_decay"],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": params["embeddings"],
                            "lr": self.args.learning_rate_embedding_recovery,
                            "weight_decay": self.args.weight_decay,
                        },
                    ]
            elif self.args.learning_rate_time_embedding is not None:   #only used for finetuning..
                time_embedding_params = self.get_conditional_norm_params(self.model)
                params = {"standard": [], "no_weight_decay": [], "time_embedding": []}
                for n, p in opt_model.named_parameters():
                    if n in decay_parameters and p.requires_grad:
                        params["standard"].append(p)
                    elif p.requires_grad:
                        if n in time_embedding_params:
                            params["time_embedding"].append(p)
                        else:
                            params["no_weight_decay"].append(p)
                optimizer_grouped_parameters = [
                    {
                        "params": params["standard"],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": params["no_weight_decay"],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": params["time_embedding"],
                        "lr": self.args.learning_rate_time_embedding,
                        "weight_decay": 0.0,
                    },
                ]
            else: #some parameters are not decayed
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            if optimizer_cls.__name__ == "Adam8bit": #AdamW is the name
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        print(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(
                            f"bitsandbytes: will optimize {module} in fp32"
                        )
                print(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def set_ar_steps(self, ar_steps=None, output_all_steps=False): ##custom function, not inside transformers library
        self.ar_steps = ar_steps
        if self.ar_steps is not None and output_all_steps:
            self.output_all_steps = True

    def _model_forward(self, model, inputs):  ##custom function, not inside transformers library
        
        ##When training with autoregressive rollouts
        if self.ar_steps is not None and model.config.use_conditioning: #Here conditioning means 'time_conditioning' which is true for both CE-RP and PoissonGauss dataset
            channel_difference = (
                model.config.num_channels > model.config.num_out_channels) #Here we assume that the channel not predicted in the output is the last channel in the input (like Re or Ma).
            # TODO: if outputs is not a dataclass this will break
            ## inputs.keys() = dict_keys(['pixel_values', 'labels', 'time', 'pixel_mask'])
            ## inputs['pixel_values'].shape = torch.Size([40, 4, 128, 128]) for CERP pretrain
            ## inputs['labels'].shape = torch.Size([40, 4, 128, 128])
            ## inputs['time'].shape = torch.Size([40])
            ## inputs['pixel_mask'].shape = torch.Size([40, 4])
            if isinstance(self.ar_steps, int): 
                inputs = {**inputs, **{"time": inputs["time"] / self.ar_steps}} 
                if self.output_all_steps:
                    loss_ = []
                    outputs_ = []
                    hidden_states_ = []
                    attentions_ = []
                    reshaped_hidden_states_ = []
                else:
                    loss = 0
                for i in range(self.ar_steps):
                    outputs = model(**inputs)
                    #outputs is a "ScOTOutput" object which is inherited from ModelOutput
                    #outputs.__dict__.keys() = dict_keys(['loss', 'output', 'hidden_states', 'attentions', 'reshaped_hidden_states'])
                    #outputs.__dict__['output'].shape =torch.Size([16, 4, 128, 128])
                    #outputs.__dict__['loss'] is a scalar tensor
                    #hidden_states': None, 'attentions': None, 'reshaped_hidden_states': None
                    if self.output_all_steps:
                        outputs_.append(outputs.output.detach())
                        if outputs.hidden_states is not None:
                            hidden_states_.append(outputs.hidden_states)
                        if outputs.attentions is not None:
                            attentions_.append(outputs.attentions)
                        if outputs.reshaped_hidden_states is not None:
                            reshaped_hidden_states_.append(
                                outputs.reshaped_hidden_states
                            )
                        if outputs.loss is not None:
                            loss_.append(outputs.loss)
                    else:
                        if outputs.loss is not None:
                            loss += outputs.loss #loss is added up across all ar_steps and we obtain a scalar. This is divided by ar_steps
                    
                    #recreate the inputs to be fed to the model for the next step
                    inputs = {
                        **inputs,
                        **{ #this part replaces the pixel_values of input with the output of the model. So the new input is the output from the previous step.
                            "pixel_values": (
                                outputs.output.detach()
                                if not channel_difference
                                else torch.cat(
                                    [
                                        outputs.output.detach(),
                                        inputs["pixel_values"][:,model.config.num_out_channels :,],
                                    ],
                                    dim=1,
                                )
                            )
                        },
                    }
                   
                if self.output_all_steps:
                    outputs.output = torch.stack(outputs_, dim=1)
                    if len(loss_) > 0:
                        outputs.loss = torch.stack(loss_, dim=0) #this is a tensor of shape (num_ar_steps,)
                        outputs.loss = outputs.loss.mean() #extra added line to make it a scalar (by me)
                    if len(hidden_states_) > 0:
                        outputs.hidden_states = [
                            torch.stack(hs, dim=1) for hs in zip(*hidden_states_)
                        ]
                    if len(attentions_) > 0:
                        outputs.attentions = [
                            torch.stack(att, dim=1) for att in zip(*attentions_)
                        ]
                    if len(reshaped_hidden_states_) > 0:
                        outputs.reshaped_hidden_states = [
                            torch.stack(rhs, dim=1)
                            for rhs in zip(*reshaped_hidden_states_)
                        ]
                else:
                    loss /= self.ar_steps #take the mean of the loss across all ar_steps
                    outputs.loss = loss
            
            #In-homogeneous rollout (#not sure if this makes any sense)
            elif isinstance(self.ar_steps, list):
                if self.output_all_steps:
                    loss_ = []
                    outputs_ = []
                    hidden_states_ = []
                    attentions_ = []
                    reshaped_hidden_states_ = []
                else:
                    loss = 0
                lead_time = inputs["time"]
                for i in self.ar_steps:
                    inputs = {
                        **inputs,
                        **{"time": lead_time * i},
                    }
                    outputs = model(**inputs)
                    if self.output_all_steps:
                        outputs_.append(outputs.output.detach())
                    if self.output_all_steps:
                        outputs_.append(outputs.output.detach())
                        if outputs.hidden_states is not None:
                            hidden_states_.append(outputs.hidden_states)
                        if outputs.attentions is not None:
                            attentions_.append(outputs.attentions)
                        if outputs.reshaped_hidden_states is not None:
                            reshaped_hidden_states_.append(
                                outputs.reshaped_hidden_states
                            )
                        if outputs.loss is not None:
                            loss_.append(outputs.loss)
                    else:
                        if outputs.loss is not None:
                            loss += outputs.loss
                    inputs = {
                        **inputs,
                        **{
                            "pixel_values": (
                                outputs.output.detach()
                                if not channel_difference
                                else torch.cat(
                                    [
                                        outputs.output.detach(),
                                        inputs["pixel_values"][
                                            :,
                                            model.config.num_out_channels :,
                                        ],
                                    ],
                                    dim=1,
                                )
                            )
                        },
                    }
                if self.output_all_steps:
                    outputs.output = torch.stack(outputs_, dim=1)
                    if len(loss_) > 0:
                        outputs.loss = torch.stack(loss_, dim=1)
                    if len(hidden_states_) > 0:
                        outputs.hidden_states = [
                            torch.stack(hs, dim=1) for hs in zip(*hidden_states_)
                        ]
                    if len(attentions_) > 0:
                        outputs.attentions = [
                            torch.stack(att, dim=1) for att in zip(*attentions_)
                        ]
                    if len(reshaped_hidden_states_) > 0:
                        outputs.reshaped_hidden_states = [
                            torch.stack(rhs, dim=1)
                            for rhs in zip(*reshaped_hidden_states_)
                        ]
                else:
                    loss /= len(self.ar_steps)
                    outputs.loss = loss
            else:
                raise ValueError(
                    "num_ar_steps must be an integer or a list of integers."
                )
        
        # No autoregressive rollouts 
        else:
            outputs = model(**inputs)

        return outputs

    def compute_loss(self, model, inputs, return_outputs=False):   ##overrides the one in the  base class from transformers library
                                                                #TODO 
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = self._model_forward(model, inputs) #outputs[0] has the loss
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    
    def prediction_step( ##overrides the one in the  base class from transformers library (only one single line is changed)
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels: #has_labels is true for validation and testing
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled(): #doesnt go here by default, this is for distributed inference
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(
                            v
                            for k, v in raw_outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else: #not sure why this 'else' is needed.
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(
                            v for k, v in raw_outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels: #enters here (for both one step and autoregressive rollouts)
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss( #this has the _model_perdict() function inside it
                            model, inputs, return_outputs=True
                        ) 
                    loss = loss.mean().detach() #mean() is used when: self.output_all_steps = True which results in loss being a tensor of shape (num_ar_steps,) and we take the mean

                    if isinstance(outputs, dict): #enters here (outputs.keys() = dict_keys(['loss', 'output', 'hidden_states', 'attentions', 'reshaped_hidden_states']))
                        logits = tuple( #saves the outputs['output]
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]# ignores the keys
                        ) #logits is a tuple of the outputs['output'], for CE-RP it has only one element with logits[0] = outputs['output'] 
                          #and logits[0].shape = torch.Size([B, 4, 128, 128])
                    else: #if outputs is a tensor, then logits is the slice corresponding to outputs[1:] as the 0th index is the loss
                        logits = outputs[1:]
                else: ##not sure why this 'else' is needed.
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = self._model_forward(model, inputs) #this is the only line which is different from the base class
                        ##in the base class it is outputs = model(**inputs),but since we have the autoregressive code as well, we need to use the model_forward function
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v for k, v in outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0: #self.args.past_index = -1 by default
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only: #prediction_loss_only is false by default
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1: #this is true for CE-RP, logits[0] = outputs['output'] has shape torch.Size([B, 4, 128, 128])
            logits = logits[0] #extract the output from the tuple

        return (loss, logits, labels)
