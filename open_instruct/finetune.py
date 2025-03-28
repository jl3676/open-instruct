# !/usr/bin/env python
# coding=utf-8
# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import json
import random
from datetime import timedelta
from functools import partial

import datasets
import deepspeed
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    GPT2Tokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    OPTForCausalLM,
    get_scheduler,
)

from utils import ArgumentParserPlus, FlatArguments

logger = get_logger(__name__)


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length, add_bos=False):
    """
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    """
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example["prompt"].endswith((" ", "\n", "\t")) and not example["completion"].startswith((" ", "\n", "\t")):
        example_text = example["prompt"] + " " + example["completion"]
    else:
        example_text = example["prompt"] + example["completion"]
    example_text = example_text + tokenizer.eos_token
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example["prompt"], return_tensors="pt", max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, : tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length, analysis_type, add_bos=False, static_prompts_file='../src/static_prompts.json'):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(messages):
        with open(static_prompts_file, 'r') as file:
            static_prompts = json.load(file)

        harm_extent_minor = static_prompts['harm_extent_minor']
        harm_extent_significant = static_prompts['harm_extent_significant']
        harm_extent_substantial = static_prompts['harm_extent_substantial']
        harm_extent_major = static_prompts['harm_extent_major']

        benefit_extent_minor = static_prompts['benefit_extent_minor']
        benefit_extent_significant = static_prompts['benefit_extent_significant']
        benefit_extent_substantial = static_prompts['benefit_extent_substantial']
        benefit_extent_major = static_prompts['benefit_extent_major']

        likelihood_low = static_prompts['likelihood_low']
        likelihood_medium = static_prompts['likelihood_medium']
        likelihood_high = static_prompts['likelihood_high']
        
        finetuning_setup = static_prompts['finetuning_setup']
        finetuning_setup_harms = static_prompts['finetuning_setup_harms']
        finetuning_setup_benefits = static_prompts['finetuning_setup_benefits']
        finetuning_JSON_format = static_prompts['finetuning_JSON_format']
        finetuning_JSON_format_harms = static_prompts['finetuning_JSON_format_harms']
        finetuning_JSON_format_benefits = static_prompts['finetuning_JSON_format_benefits']
        AIR2024_taxonomy = static_prompts['AIR2024_taxonomy_min']
        harmful_effects_taxonomy = static_prompts['harmful_effects_taxonomy_min']
        beneficial_effects_taxonomy = static_prompts['beneficial_effects_taxonomy_min']

        message_text = ""
        for message in messages:
            if message["role"] == "system":
                if 'harm' in analysis_type:
                    message_text += """<|system|>
Given a scenario based on an interaction between an AI language model and its user, who has the capabilities of an everyday person, analyze the harms of the scenario on all stakeholders without additional explanations in plain text. For each stakeholder, analyze the potential harmful actions that may be performed in the scenario or as a result of the scenario that may harm the stakeholder. For each action, analyze the harmful effects it may incur on the stakeholder. For each effect, analyze the immediacy, extent, and likelihood of harm it may cause on the stakeholder. Return your output in a JSON format defined below for each entry. *description* contains a description of the information you need to fill in at the corresponding place. Only return the JSON string. Do not provide any information or explanation besides what is specified in the JSON format.

The JSON format is:
[{"stakeholder": *fill in with an individual, group, community, or entity in society that may be affected as a result of this scenario*, "harms": [{"action": *fill in with a type of harmful action defined in the given hierarchical taxonomy of harmful action types that may be performed in the scenario or as a result of the scenario that may harm this stakeholder in the form of ##high-level category name## #mid-level category name# low-level category name*, "effects": [{"effect": *fill in with the number and name of a harmful effect category defined in the given taxonomy of harmful effects that this harmful action may cause on this stakeholder given this scenario*, "immediacy": *true if the harmful effect is short-term or immediate, and false if the harmful effect is long-term or downstream*, "extent": *the extent to which the stakeholder may be harmed by this harmful effect (minor, significant, substantial, or major, as defined below)*, "likelihood": *the likelihood that the stakeholder may be harmed by this harmful effect (low, medium, or high, as defined below)*}, ... *List as many distinct and valid harmful effects as you can*]}, ... *List as many more distinct and valid harmful actions as you can*]}, ... *list as many distinct and valid stakeholders as you can*]

The hierarchical taxonomy of harmful action types is defined as:
##System and Operational Risks##
#Security Risks#
1. Confidentiality
2. Integrity
3. Availability
#Operational Misuses#
4. Automated Decision-Making
5. Autonomous Unsafe Operation of Systems
6. Advice in Heavily Regulated Industries
##Content Safety Risks##
#Violence & Extremism#
7. Supporting Malicious Organized Groups
8. Celebrating Suffering
9. Violent Acts
10. Depicting Violence
11. Weapon Usage & Development
12. Military and Warfare
#Hate/Toxicity#
13. Harassment
14. Hate Speech, Inciting/Promoting/Expressing hatred
15. Perpetuating Harmful Beliefs
16. Offensive Language
#Sexual Content#
17. Adult Content
18. Erotic
19. Non-Consensual Nudity
20. Monetized
#Child Harm#
21. Endangerment, Harm, or Abuse of Children
22. Child Sexual Abuse
#Self-harm#
23. Suicidal and Non-suicidal Self-injury
#Political Usage#
24. Political Persuasion
25. Influencing Politics
26. Deterring Democratic Participation
27. Disrupting Social Order
#Economic Harm#
28. High-Risk Financial Activities
29. Unfair Market Practices
30. Disempowering Workers
31. Fraudulent Schemes
#Deception#
32. Fraud
33. Academic Dishonesty
34. Mis/disinformation
#Manipulation#
35. Sowing Division
36. Misrepresentation
#Defamation#
37. Types of Defamation
##Legal and Rights-Related Risks##
#Fundamental Rights#
38. Violating Specific Types of Rights
#Discrimination/Bias#
39. Discriminatory Activities
40. Protected Characteristics
#Privacy#
41. Unauthorized Privacy Violations
42. Types of Sensitive Data
#Criminal Activities#
43. Illegal/Regulated Substances
44. Illegal Services/Exploitation
45. Other Unlawful/Criminal Activities

The taxonomy of harmful effects is defined as:
#Physical harm#
1. Death
2. Physical health risks
#Psychological harm#
3. Transient emotional changes
4. Persistent emotional state changes
#Social harm#
5. Social deficits
#Property harm#
6. Physical property loss
7. Financial property loss
8. Intellectual property infringement
#Liberty harm#
9. Freedom and agency limitations
10. Restricting, requiring, or degrading accurate information access
#Collective harm#
11. Institutional harm
12. Public resource damage
13. Societal disruption
#Ecological harm#
14. Harm to non-human animals
15. Ecosystem damage

The extents of harm are defined as:
Minor: harmful consequences are isolated, reversible, temporary, treatable, and do not require significant intervention or resources to mitigate; the stakeholder may experience limited interruption of functional ability and/or quality of life.
Significant: harmful consequences are endemic, reversible but with significant effort, acute, treatable with significant intervention or resources; the stakeholder may experience significant interference of functional ability and/or quality of life.
Substantial: harmful consequences are widespread, not easily reversible-if at all, chronic, and not treatable even with significant intervention or resources; the stakeholder may experience substantial impairment of functional ability and/or quality of life.
Major: harmful consequences are systemic, irreversible, persistent, critical, catastrophic, and imminently threatening; the stakeholder may experience destruction of functional ability and/or quality of life.

The likelihoods of harms are defined as:
Low: unlikely to occur, but not impossible. Rare under normal circumstances; less than 30 percent chance of occurring.
Medium: possible occurrence, might happen under certain conditions or occasionally; between 30 and 70 percent chance of occurring.
High: likely to occur, happens frequently or under normal conditions; above 70 percent chance of occurring.


"""
                    # message_text += "<|system|>\n" + finetuning_setup_harms + \
                    #     "The JSON format is:\n" + finetuning_JSON_format_harms + \
                    #     "The hierarchical taxonomy of harmful action types is defined as:\n" + AIR2024_taxonomy + \
                    #     "The taxonomy of harmful effects is defined as:\n" + harmful_effects_taxonomy + \
                    #     "The extents of harm are defined as:\n" + harm_extent_minor + harm_extent_significant + harm_extent_substantial + harm_extent_major + \
                    #     "The likelihoods of harms are defined as:\n" + likelihood_low + likelihood_medium + likelihood_high.strip() + "\n"
                elif 'benefit' in analysis_type:
                    message_text += """<|system|>
Given a scenario based on an interaction between an AI language model and its user, who has the capabilities of an everyday person, analyze the benefits of the scenario on all stakeholders without additional explanations in plain text. For each stakeholder, analyze the potential beneficial actions that may be performed in the scenario or as a result of the scenario that may benefit the stakeholder. For each action, analyze the beneficial effects it may incur on the stakeholder. For each effect, analyze the immediacy, extent, and likelihood of benefit it may cause on the stakeholder. Return your output in a JSON format defined below for each entry. *description* contains a description of the information you need to fill in at the corresponding place. Only return the JSON string. Do not provide any information or explanation besides what is specified in the JSON format.

The JSON format is:
[{"stakeholder": *fill in with an individual, group, community, or entity in society that may be affected as a result of this scenario*,"benefits": [{"action": *fill in with a beneficial action that may be performed in the scenario or as a result of the scenario that may benefit this stakeholder*, "effects": [{"effect": *fill in with the number and name of a beneficial effect category defined in the given taxonomy of beneficial effects that this beneficial action may cause on this stakeholder given this scenario*, "immediacy": *true if the beneficial effect is short-term or immediate, and false if the beneficial effect is long-term or downstream*, "extent": *the extent to which the stakeholder may benefit from this beneficial effect (minor, significant, substantial, or major, as defined below)*,"likelihood": *the likelihood that the stakeholder may benefit from this beneficial effect (low, medium, or high, as defined below)*}, ... *List as many distinct and valid beneficial effects as you can*]}, ... *List as many more distinct and valid beneficial actions as you can*]}, ... *list as many distinct and valid stakeholders as you can*]

The taxonomy of beneficial effects is defined as:
#Physical benefit#
1. Prevention of death
2. Physical health benefits
#Psychological benefit#
3. Transient emotional changes
4. Persistent emotional state changes
#Social benefit#
5. Social gains
#Property benefit#
6. Physical property gains
7. Financial property gains
8. Intellectual property gains
#Liberty benefit#
9. Freedom and agency benefits
10. Gain of accurate information access
#Collective benefit#
11. Institutional benefits
12. Public resource gains
13. Societal order maintenance
#Ecological benefit#
14. Benefits to non-human animals
15. Ecosystem benefits

The extents of benefit are defined as:
Minor: beneficial consequences are isolated, temporary, noticeable but have limited impact on the overall situation or outcome; the stakeholder may experience a limited enhancement of functional ability and/or quality of life.
Significant: beneficial consequences are widespread, significant, noticeable, and can lead to meaningful improvements and tangible impacts; the stakeholder may experience significant improvement in functional ability and/or quality of life.
Substantial: beneficial consequences are extensive, important, and can lead to considerable positive changes and profound impact on the situation or outcome; the stakeholder may experience substantial enhancement of functional ability and/or quality of life.
Major: beneficial consequences are systemic, persistent, critical, highly impactful, and can lead to transformative changes that significantly alter the courses of events; the stakeholder may experience a profound improvement in functional ability and/or quality of life.

The likelihoods of benefits are defined as:
Low: unlikely to occur, but not impossible. Rare under normal circumstances; less than 30 percent chance of occurring.
Medium: possible occurrence, might happen under certain conditions or occasionally; between 30 and 70 percent chance of occurring.
High: likely to occur, happens frequently or under normal conditions; above 70 percent chance of occurring.


"""
                    # message_text += "<|system|>\n" + finetuning_setup_benefits + \
                    #     "The JSON format is:\n" + finetuning_JSON_format_benefits + \
                    #     "The taxonomy of beneficial effects is defined as:\n" + beneficial_effects_taxonomy + \
                    #     "The extents of benefit are defined as:\n" + benefit_extent_minor + benefit_extent_significant + benefit_extent_substantial + benefit_extent_major + \
                    #     "The likelihoods of benefits are defined as:\n" + likelihood_low + likelihood_medium + likelihood_high.strip() + "\n"
                else:
                    message_text += "<|system|>\n" + finetuning_setup + \
                        "The JSON format is:\n" + finetuning_JSON_format + \
                        "The hierarchical taxonomy of harmful action types is defined as:\n" + AIR2024_taxonomy + \
                        "The taxonomy of harmful effects is defined as:\n" + harmful_effects_taxonomy + \
                        "The taxonomy of beneficial effects is defined as:\n" + beneficial_effects_taxonomy + \
                        "The extents of harm are defined as:\n" + harm_extent_minor + harm_extent_significant + harm_extent_substantial + harm_extent_major + \
                        "The extents of benefit are defined as:\n" + benefit_extent_minor + benefit_extent_significant + benefit_extent_substantial + benefit_extent_major + \
                        "The likelihoods of harms and benefits are defined as:\n" + likelihood_low + likelihood_medium + likelihood_high.strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    with open('temp.txt', 'a') as f:
        f.write(str(input_ids.shape[-1]) + '\n')
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    messages_so_far = example_text.split("<|assistant|>\n")[0]
    messages_len = tokenizer(messages_so_far, return_tensors="pt", max_length=max_seq_length, truncation=True).input_ids.shape[1]
    labels[:, :messages_len] = -100
    # for message_idx, message in enumerate(messages):
    #     if message["role"] != "assistant":
    #         if message_idx == 0:
    #             message_start_idx = 0
    #         else:
    #             message_start_idx = tokenizer(
    #                 _concat_messages(messages[:message_idx]),
    #                 return_tensors="pt",
    #                 max_length=max_seq_length,
    #                 truncation=True,
    #             ).input_ids.shape[1]
    #         if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
    #             # here we also ignore the role of the assistant
    #             messages_so_far = _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
    #         else:
    #             messages_so_far = _concat_messages(messages[: message_idx + 1])
    #         message_end_idx = tokenizer(
    #             messages_so_far, return_tensors="pt", max_length=max_seq_length, truncation=True
    #         ).input_ids.shape[1]
    #         labels[:, message_start_idx:message_end_idx] = -100

    #         if message_end_idx >= max_seq_length:
    #             break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def save_with_accelerate(accelerator, model, tokenizer, output_dir, args):
    # set the generation config to an empty setting to be safe.
    # we usually do greedy decoding for generation, so this should be okay.
    # otherwise, we get an error thrown at save time.
    model.generation_config = transformers.GenerationConfig(
        temperature=None, top_p=None, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id
    )

    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    if args.use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        # don't use safetensors for saving for now
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False,
        )


def main():
    parser = ArgumentParserPlus((FlatArguments))
    args = parser.parse()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
            token=os.getenv("HF_TOKEN", None),
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
            token=os.getenv("HF_TOKEN", None),
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    tokenizer_revision = args.model_revision if args.tokenizer_revision is None else args.tokenizer_revision

    if tokenizer_revision != args.model_revision:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tokenizer_revision}` is different
                   from the model revision `{args.model_revision}`."""
        logger.warn(warning)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision,
            token=os.getenv("HF_TOKEN", None),
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision,
            token=os.getenv("HF_TOKEN", None),
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index}  # force data-parallel training.
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True if args.use_flash_attn else False,
                revision=args.model_revision,
                token=os.getenv("HF_TOKEN", None),
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=args.trust_remote_code,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                use_flash_attention_2=True if args.use_flash_attn else False,
                revision=args.model_revision,
                token=os.getenv("HF_TOKEN", None),
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )
        assert num_added_tokens in [
            0,
            1,
        ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        # OLMo newer models use this tokenizer
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
            assert (
                args.add_bos
            ), "For OLMo with GPTNeoX, you must add bos token to the beginning of the input sequence."
        # else, pythia / other models
        else:
            num_added_tokens = tokenizer.add_special_tokens(
                {
                    "pad_token": "<pad>",
                }
            )
            assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({"unk_token": "<unk>"})
    elif isinstance(tokenizer, transformers.PreTrainedTokenizerFast) and tokenizer.pad_token is None:
        num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
        assert num_added_tokens == 1, "We detected no padding token but add_special_tokens did not add one."

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # update embedding size after resizing for sum loss
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    # add dummy tokens to the tokenizer to match the size of the embeddings
    print(f"Adding {embedding_size - len(tokenizer)} dummy tokens to the tokenizer.")
    for i in range(embedding_size - len(tokenizer)):
        tokenizer.add_tokens([f"dummy_id_{i}"])

    # set the tokenizer chat template to the tulu format
    # this makes evaluation/etc easier down the line.
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"  # noqa: E501
    if args.add_bos:
        # also add bos in the chat template
        tokenizer.chat_template = "{{ bos_token }}" + tokenizer.chat_template

    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Preprocessing the datasets.
    if "prompt" in raw_datasets["train"].column_names and "completion" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos=args.add_bos,
        )
    elif "messages" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos=args.add_bos,
            analysis_type='harms' if 'benefit' not in args.train_file else ('benefit' if 'harm' not in args.train_file else 'both'),
        )
    else:
        raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")

    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[
                name
                for name in raw_datasets["train"].column_names
                if name not in ["input_ids", "labels", "attention_mask"]
            ],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(lambda example: (example["labels"] != -100).any())

    train_dataset = lm_datasets["train"]
    # debugging tool for fewer samples
    if args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.max_train_samples)
        logger.info(f"Limiting training samples to {max_train_samples} from {len(train_dataset)}.")
        train_dataset = train_dataset.select(range(max_train_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.use_qlora:
        from bitsandbytes.optim import AdamW

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True,
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler
    # for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set.
    # In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set.
    # So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using
    # the entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number
    # of updates matches the num_training_steps.
    num_training_steps_for_scheduler = (
        args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )
    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
        accelerator.init_trackers(
            "open_instruct_sft", experiment_config, init_kwargs={"wandb": {"entity": args.wandb_entity}}
        )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                if args.reduce_loss == "mean":
                    loss = outputs.loss
                else:
                    # reduce loss is sum
                    # this ensures that we weight all tokens in the dataset equally,
                    # rather than weighting each overall example equally when
                    # using high amounts of gradient accumulation.
                    # this can result in > 5 point improvements in AlpacaEval
                    # see https://github.com/huggingface/transformers/issues/24725 for
                    # more discussion and details.
                    logits = outputs.logits
                    labels = batch["labels"]
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
                    shift_logits = shift_logits.view(-1, embedding_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = (
                            accelerator.gather(total_loss).mean().item()
                            / args.gradient_accumulation_steps
                            / args.logging_steps
                        )
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

    if args.output_dir is not None:
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        save_with_accelerate(accelerator, model, tokenizer, args.output_dir, args)

    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()
