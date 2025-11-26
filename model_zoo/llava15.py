import re
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import random
from transformers import AutoProcessor, LlamaTokenizerFast, CLIPImageProcessor
import pdb
# import probe_llava
from .llava import  LlavaForConditionalGeneration, LlavaForConditionalGenerationScal

import torch
import torch.nn.functional as F
from PIL import Image
import requests
import json
import os
from collections import Counter
# from model_zoo.utils import normalize_answer,chat_completion_request,run_conversation

from PIL import Image
import math
MODEL='llava-hf/llava-1.5-7b-hf'

import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput, SampleDecoderOnlyOutput, SampleEncoderDecoderOutput,GenerateEncoderDecoderOutput,GenerateDecoderOnlyOutput,GenerateNonBeamOutput
import os
import json
import random
import numpy as np
import torch
from tqdm import tqdm
def _add_weight_greedy_search(
    self,
    input_ids: torch. LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    output_logits: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    # keys:Optional[torch.Tensor] = None,
    weight: Optional[float] = None,
    adjust_method: Optional[str] = None,
    pos: Optional[torch.Tensor] = None,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    scores = () if (return_dict_in_generate and output_scores) else None
    before = () if (return_dict_in_generate) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    if "inputs_embeds" in model_kwargs:
        cur_len = model_kwargs["inputs_embeds"].shape[1]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
    
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        import pdb
        # 
        if 'Scal' not in str(type(self)):
            outputs = self(
                **model_inputs,
               
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        else:
            
            outputs = self(
                **model_inputs,
                weight=weight,
                adjust_method=adjust_method,
                pos=pos,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids
    
def change_greedy_to_add_weight():
    transformers.generation.utils.GenerationMixin._greedy_search = _add_weight_greedy_search

def parse_objects(response):
    """'{obj1} and {obj2}' 형태의 답변에서 obj1, obj2 추출"""
    response = response.strip().strip("'\"")
    
    if " and " in response.lower():
        parts = re.split(r'\s+and\s+', response, flags=re.IGNORECASE)
        if len(parts) >= 2:
            obj1 = parts[0].strip().lower()
            obj2 = parts[1].strip()
            if obj2.endswith('.'): 
                obj2 = obj2[:-1]
            return obj1, obj2
    
    return None, None

def parse_relation_type(response):
    """Relation type 추출"""
    response_lower = response.lower().strip().strip("'\"")
    
    # 정확한 매칭
    if response_lower in ['left-right', 'leftright', 'left right']:
        return 'left-right'
    elif response_lower in ['front-back', 'frontback', 'front back', 'depth']:
        return 'front-back'
    elif response_lower in ['top-bottom', 'topbottom', 'top bottom', 'vertical']:
        return 'top-bottom'
    elif response_lower in ['diagonal']:
        return 'diagonal'
    
    # 부분 매칭
    if 'left' in response_lower and 'right' in response_lower:
        return 'left-right'
    elif 'front' in response_lower or 'back' in response_lower or 'behind' in response_lower:
        return 'front-back'
    elif 'top' in response_lower or 'bottom' in response_lower:
        return 'top-bottom'
    
    return None

def check_logical_consistency(response_a, response_b):
    """양방향 답변의 논리적 일관성 검증"""
    # 기본 symmetric pairs
    symmetric_pairs = {
        'in front of': 'behind',
        'behind': 'in front of',
        'on': 'under',
        'under': 'on'
    }
    
    # Left/Right 관련 유연한 매칭을 위한 그룹
    left_variations = ['left of', 'left']
    right_variations = ['right of', 'right']
    
    resp_a = response_a.strip().strip("'\"").lower()
    resp_b = response_b.strip().strip("'\"").lower()
    
    # Left/Right 그룹 간 대칭성 체크
    if resp_a in left_variations and resp_b in right_variations:
        return True
    if resp_a in right_variations and resp_b in left_variations:
        return True
    
    # 기타 symmetric pairs 체크
    for key, value in symmetric_pairs.items():
        if key == resp_a and value == resp_b:
            return True
        if value == resp_a and key == resp_b:
            return True
    
    return False

def get_symmetric_relation(relation):
    """주어진 관계의 대칭 관계 반환"""
    symmetric_pairs = {
        'Left': 'Right of or Right',
        'Left of': 'Right of or Right',
        'Right': 'Left of or Left',
        'Right of': 'Left of or Left',
        'In front of': 'Behind',
        'Behind': 'In front of',
        'On': 'Under',
        'Under': 'On'
    }
    
    relation_normalized = relation.strip().strip("'\"")
    return symmetric_pairs.get(relation_normalized, "Unknown")

class LlavaWrapper:
    def __init__(self, root_dir, device,method):
        
        if method=='scaling_vis' or method=='adapt_vis' or method=='adapt_vis_2' or method=='adaptvis_bidirectional':
            self.model = LlavaForConditionalGenerationScal.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir,ignore_mismatched_sizes=True).eval().to(device)

        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(MODEL, revision='a272c74', cache_dir=root_dir,ignore_mismatched_sizes=True).eval().to(device)

        self.feature_extractor = CLIPImageProcessor.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir)
        self.tokenizer = LlamaTokenizerFast.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir)
        self.processor = AutoProcessor.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir)

        self.device = device
    
    def _kl(self, p, q):
        p = p + 1e-12
        q = q + 1e-12
        p = p / p.sum()
        q = q / q.sum()
        return torch.sum(p * torch.log(p / q))
    
    def get_uncertainty(self, prob, method=None):
        if method=='kl_divergence':
            options = ["Left", "Right", "On", "Under"]
            option_ids = [self.tokenizer.encode(r, add_special_tokens=False)[0] for r in options]
            prob_options = prob[0, option_ids]
            uniform_dist = torch.ones_like(prob_options) / len(option_ids)
            print(prob_options, uniform_dist)
            return float(self._kl(prob_options, uniform_dist).item())
        else:
            return np.round(float(max(torch.nn.functional.softmax(prob, dim=-1)[0])), 2)
    
    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=64, normalize=False):
        num_text = len(texts)
        text_embeds = []
        for i in tqdm(range(0, num_text, text_batch_size)):
            text = texts[i: min(num_text, i+text_batch_size)]
            text_input = self.tokenizer(text=text, return_tensors="pt", padding="max_length", max_length=77).to(self.device)
            text_feats = self.model.llava.get_text_features(**text_input).cpu().numpy()[:, 0, :].to(self.device)
            if normalize:
                text_feats = text_feats / np.linalg.norm(text_feats, axis=1, keepdims=True)          
            text_embeds.append(text_feats)   
            
        return np.concatenate(text_embeds, axis=0)
    
    @torch.no_grad()
    def get_image_embeddings(self, image_loader, normalize=False):
        image_embeds = []
        for batch in tqdm(image_loader):
            images = batch["image"]
            inputs = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
            image_feats = self.model.llava.get_image_features(**inputs).cpu().numpy()[:, 0, :]
            if normalize:
                image_feats = image_feats / np.linalg.norm(image_feats, axis=1, keepdims=True)
            image_embeds.append(image_feats)

        return np.concatenate(image_embeds, axis=0)
    
    def get_retrieval_scores_dataset(self, loader):
        texts = loader.dataset.text
        text_embeds = self.get_text_embeddings(texts, normalize=True)
        image_embeds = self.get_image_embeddings(loader, normalize=True)
        scores = image_embeds @ text_embeds.T
        return scores
    
    @torch.no_grad()
    def get_out_scores_wh_batched(self, dataset, joint_loader, method, weight, option, threshold, weight1, weight2):
        scores = []  # To store scores for each batch
        index_of_total = 0  # Track total number of prompts processed
        acc = 0  # Track the number of correct predictions
        correct_id = []  # Track indices of correct predictions

        # Determine the correct question-answer file based on the dataset
        qst_ans_file = f'prompts/{dataset}_with_answer_{option}_options.jsonl'
        
        # Load prompts and answers from the question-answer file
        with open(qst_ans_file, 'r') as file:
            prompt_list = []
            answer_list = []
            first_prompt_list = []
            second_prompt_list = []
            for line in file:
                data = json.loads(line)
                # Select prompt based on mode
                
                prompt_list.append(data["question"])
                
                # Store additional prompts if adjustment method is 'sub'
                
                answer_list.append(data["answer"])

        # Sampling configuration
        SAMPLE = True
        TEST = os.getenv('TEST_MODE', 'False') == 'True'
        total_data_count = len(prompt_list)
        
        # Perform sampling if enabled
        if SAMPLE:
            idx_file_path = f'./output/sampled_idx_{dataset}.npy'
            
            if os.path.exists(idx_file_path):
                sampled_indices = np.load(idx_file_path).tolist()
            else:
                sampled_indices = random.sample(range(total_data_count), int(0.2 * total_data_count))
                sampled_indices.sort()
                np.save(idx_file_path, np.array(sampled_indices))

            # For testing mode, use unsampled indices
            if TEST:
                all_indices = set(range(total_data_count))
                unsampled_indices = list(all_indices - set(sampled_indices))
                unsampled_indices.sort()
                sampled_indices = unsampled_indices

            # Subset prompts and answers based on sampled indices
            prompt_list = [prompt_list[i] for i in sampled_indices]
            answer_list = [answer_list[i] for i in sampled_indices]

        # Create directory for saving attention maps
        save_attn_dir = f"./output/{dataset}_weight{weight:.2f}"
        os.makedirs(save_attn_dir, exist_ok=True)

        results = []  # Store results for each generated sequence
        reasoning_chains = []
        for batch in tqdm(joint_loader):
            batch_scores = []
            
            # Set environment variable for attention map save path
            os.environ['SAVE_ATTN_PATH'] = f'{save_attn_dir}/{index_of_total}/'
            os.makedirs(os.environ['SAVE_ATTN_PATH'], exist_ok=True)

            # Iterate over each image option in the batch
            for i_option in batch["image_options"]:
                im_scores = []
                uncertainty_prob = None
                uncertainty_kl = None
                for _ in i_option:
                    prompt = prompt_list[index_of_total]
                    
                    # Preprocess input for the model
                    single_input = self.processor(
                        text=prompt, images=_, padding="max_length", return_tensors="pt", max_length=77
                    ).to(self.device)
                    
                    # Create key mask for special token
                    keys = [torch.where(input_id == 32001, 1, 0) for input_id in single_input['input_ids']]

                    # Generate predictions based on specified method
                    if method == 'bidirectional_reasoning':
                        # 옵션 리스트 생성
                        c_option = batch["caption_options"]
                        if len(list(c_option)) == 4:
                            option_list = ['In front of', 'Behind', 'Left of', 'Right of']
                        elif len(list(c_option)) == 2:
                            option_list = ['Yes', 'No']
                        else:
                            option_list = []
                        
                        # Bidirectional reasoning 수행
                        result = self.bidirectional_reasoning_single(_, prompt, option_list)
                        
                        gen = result['final_prediction']
                        uncertainty = result['avg_prob']
                        
                        # 결과에 추가 정보 저장
                        result_detail = {
                            "Prompt": prompt,
                            "Generation": gen,
                            "Golden": answer_list[index_of_total][0],
                            "Uncertainty": uncertainty,
                            "Is_Consistent": result['is_consistent'],
                            "Fallback": result.get('fallback', False)
                        }
                        results.append(result_detail)
                        reasoning_chains.append(result['reasoning_chain'])
                        
                    elif method == 'bidirectional_reasoning_2':
                        # 옵션 리스트 생성
                        c_option = batch["caption_options"]
                        if len(list(c_option)) == 4:
                            option_list = ['In front of', 'Behind', 'Left of', 'Right of']
                        elif len(list(c_option)) == 2:
                            option_list = ['Yes', 'No']
                        else:
                            option_list = []
                        
                        # Bidirectional reasoning 수행
                        result = self.bidirectional_reasoning_2(_, prompt, option_list)
                        
                        gen = result['final_prediction']
                        uncertainty = result['avg_prob']
                        
                        # 결과에 추가 정보 저장
                        result_detail = {
                            "Prompt": prompt,
                            "Generation": gen,
                            "Golden": answer_list[index_of_total][0],
                            "Uncertainty": uncertainty,
                            "Is_Consistent": result['is_consistent'],
                            "Fallback": result.get('fallback', False)
                        }
                        print(f"\nidx : {index_of_total}")
                        results.append(result_detail)
                        reasoning_chains.append(result['reasoning_chain'])
                        
                    elif method == 'adaptvis_bidirectional':
                        # 옵션 리스트 생성
                        c_option = batch["caption_options"]
                        if len(list(c_option)) == 4:
                            option_list = ['In front of', 'Behind', 'Left of', 'Right of']
                        elif len(list(c_option)) == 2:
                            option_list = ['Yes', 'No']
                        else:
                            option_list = []
                        
                        # AdaptVis + Bidirectional 수행 (sample_idx 전달)
                        result = self.adaptvis_bidirectional_reasoning(
                            _, prompt, option_list, threshold, weight1, weight2,
                            sample_idx=index_of_total  # ← 샘플 인덱스 전달
                        )
                        
                        gen = result['final_prediction']
                        uncertainty = result['confidence']
                        
                        # 결과 저장
                        result_detail = {
                            "Prompt": prompt,
                            "Generation": gen,
                            "Golden": answer_list[index_of_total][0],
                            "Initial_Uncertainty": result['initial_uncertainty'],
                            "Selected_Weight": result['selected_weight'],
                            "Confidence_Forward": result['confidence_forward'],
                            "Confidence_Reverse": result['confidence_reverse'],
                            "Is_Consistent": result['is_consistent'],
                            "Selection_Reason": result['selection_reason'],
                            "Final_Confidence": uncertainty
                        }
                        results.append(result_detail)
                    
                    elif method == 'scaling_vis':
                        change_greedy_to_add_weight()
                        output = self.model.generate(
                            **single_input, keys=keys, weight=weight,
                            max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                        )
                        uncertainty = np.round(float(max(torch.nn.functional.softmax(output['scores'][0], dim=-1)[0])), 2)
                        gen = self.processor.decode(output['sequences'][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True)
                    
                    elif method == 'adapt_vis':
                        change_greedy_to_add_weight()
                       
                        output = self.model.generate(
                            **single_input,weight=1.0,max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                        )
                        uncertainty_prob = self.get_uncertainty(output['scores'][0])
                        uncertainty_kl = self.get_uncertainty(output['scores'][0], method='kl_divergence')
                        print(f"\nUncertainty_prob: {uncertainty_prob}  |  Uncertainty_KL: {uncertainty_kl}  |  Threshold: {threshold}")

                        # Adjust attention based on uncertainty
                        if uncertainty_prob < threshold:
                            output = self.model.generate(
                                **single_input, keys=keys, weight=weight1, 
                                max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                            )
                        else:
                            output = self.model.generate(
                                **single_input, keys=keys, weight=weight2, 
                                max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                            )
                        gen = self.processor.decode(output['sequences'][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True)
                    
                    elif method == 'adapt_vis_2':
                        change_greedy_to_add_weight()
                       
                        output = self.model.generate(
                            **single_input,weight=1.0,max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                        )
                        uncertainty_prob = self.get_uncertainty(output['scores'][0])
                        uncertainty_kl = self.get_uncertainty(output['scores'][0], method='kl_divergence')
                        threshold = 0.0005
                        print(f"\nUncertainty_prob: {uncertainty_prob}  |  Uncertainty_KL: {uncertainty_kl}  |  Threshold: {threshold}")
                        # Adjust attention based on uncertainty
                        if uncertainty_kl < 0.00045:
                            output = self.model.generate(
                                **single_input, keys=keys, weight=weight1, 
                                max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                            )
                        elif 0.00045 <= uncertainty_kl < 0.0055:
                            pass # Using the original output
                        else:
                            output = self.model.generate(
                                **single_input, keys=keys, weight=weight2, 
                                max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                            )
                        gen = self.processor.decode(output['sequences'][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True)
                    
                    else:
                        # Default generation method
                        output = self.model.generate(
                            **single_input, max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                        )
                        gen = self.processor.decode(output['sequences'][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True)
                        uncertainty = np.round(float(max(output['scores'][0][0])), 2)

                    # Print prompt, generated text, and expected answer
                    print(f"Prompt: {prompt}\nGeneration: {gen}\nGolden: {answer_list[index_of_total][0]}")
                    
                    result = {
                        "Prompt": prompt,
                        "Generation": gen,
                        "Golden": answer_list[index_of_total][0],
                        "uncertainty_prob": uncertainty_prob,
                        "uncertainty_kl": uncertainty_kl,
                    }
                    results.append(result) if method != "bidirectional_reasoning" else None
                    
                    # Check if the generation matches the expected answer
                    c_option = batch["caption_options"]
                    if len(list(c_option)) == 4:
                        if (answer_list[index_of_total][0] in gen or answer_list[index_of_total][0].lower() in gen.lower()) \
                                and not (answer_list[index_of_total][0].lower() == 'on' and 'front' in gen.strip().lower()):
                            acc += 1
                            correct_id.append(index_of_total)
                            answers = [1, 0, 0, 0]
                        else:
                            answers = [0, 0, 1, 0]
                    
                    elif len(list(c_option)) == 2:
                        if (answer_list[index_of_total][0] in gen or answer_list[index_of_total][0].lower() in gen.lower()) \
                                and not (answer_list[index_of_total][0].lower() == 'on' and 'front' in gen.strip().lower()):
                            acc += 1
                            correct_id.append(index_of_total)
                            answers = [1, 0]
                        else:
                            answers = [0, 1]

                    im_scores.append(np.expand_dims(np.array(answers), -1))
                    index_of_total += 1

                batch_scores.append(np.concatenate(im_scores, axis=-1))

            scores.append(batch_scores)

            # Save results to file
            output_result_file_path = f'./output/results1.5_{dataset}_{method}_{weight}_{option}option_{TEST}.json'
            output_reasoning_file_path = f'./output/results1.5_{dataset}_{method}_{weight}_{option}option_{TEST}_reasoning.json'
            with open(output_result_file_path, 'w', encoding='utf-8') as fout:
                json.dump(results, fout, ensure_ascii=False, indent=4)
            print(acc, index_of_total, acc / index_of_total)
            
            with open(output_reasoning_file_path, 'w', encoding='utf-8') as f:
                json.dump(reasoning_chains, f, ensure_ascii=False, indent=4)
                            
        # Save accuracy and correct IDs to file
        print(acc / index_of_total)
        output_score_file = output_result_file_path.replace(".json", "scores.json")
        with open(output_score_file, 'w', encoding='utf-8') as fout:
            json.dump({"acc": acc / index_of_total, "correct_id": correct_id}, fout, ensure_ascii=False, indent=4)

        # Concatenate all scores and return based on dataset type
        all_scores = np.concatenate(scores, axis=0)  # N x K x L
        if dataset in ['Controlled_Images_B', 'Controlled_Images_A']:
            return (all_scores, [])
        else:
            return (acc / index_of_total, correct_id)

    @torch.no_grad()
    def get_judge_scores_vsr_batched(self, dataset, joint_loader, method, weight, threshold, weight1, weight2):
        
        
        index = 0
        TP, TN, FP, FN = 0, 0, 0, 0

        # Set the directory to save attention maps
        save_attn_dir = f"/home/user/shiqi/mmlm_mech/whatsup_vlms/output/{dataset}_weight{weight:.2f}"
        if not os.path.exists(save_attn_dir):
            print("Creating directory for saving attention maps:", save_attn_dir)
            os.makedirs(save_attn_dir)
        
        index_of_total = 0
        results = []

        # Process each batch in the joint loader
        for batch in tqdm(joint_loader):
            batch_scores = []
            
            # Create directory for saving attention maps for each batch
            os.environ['SAVE_ATTN_PATH'] = f'{save_attn_dir}/{index_of_total}/'
            os.makedirs(os.environ['SAVE_ATTN_PATH'], exist_ok=True)

            # Iterate over image options in the batch
            for i_option in batch["image_options"]:
                im_scores = []

                # Iterate over caption options
                for c_option in batch["caption_options"]:
                    prompt = "User: <image>\n Determine whether the description about the spatial relationship is correct or not. Answer with yes or no: "
                    qst = [prompt] * len(list(c_option))
                    end_fix = [" Assistant:"] * len(list(c_option))
                    concatenated_list = [s1 + s2 + s3 for s1, s2, s3 in zip(qst, c_option, end_fix)]
                    
                    # Generate responses for each concatenated input
                    for idx, text in enumerate(concatenated_list):
                        # Prepare input data for the model
                        single_input = self.processor(text=text, images=list(i_option)[idx], padding="max_length", return_tensors="pt", max_length=77).to(self.device)
                        keys = [torch.where(input_id == 32001, 1, 0) for input_id in single_input['input_ids']]
                        
                        # Apply different attention adjustment methods based on the 'method' argument
                        if method == 'scaling_vis':
                            change_greedy_to_add_weight()
                            output = self.model.generate(**single_input, keys=keys, weight=weight, max_new_tokens=100, output_scores=True, return_dict_in_generate=True)
                            uncertainty = np.round(float(max(torch.nn.functional.softmax(output['scores'][0], dim=-1)[0])), 2)
                            gen = self.processor.decode(output[0][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True, output_attentions=True)
                        
                        elif method == 'adapt_vis':
                            change_greedy_to_add_weight()
                            # Basic generation step
                            output = self.model.generate(**single_input, weight=1.0,max_new_tokens=100, output_scores=True, return_dict_in_generate=True)
                            gen = self.processor.decode(output['sequences'][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True, output_attentions=True)
                            uncertainty = np.round(float(max(output['scores'][0][0])), 2)
                            
                            # Apply weighted generation based on uncertainty
                            if uncertainty < threshold:
                                output = self.model.generate(**single_input, keys=keys, weight=weight1, max_new_tokens=100, output_scores=True, return_dict_in_generate=True)
                            else:
                                output = self.model.generate(**single_input, keys=keys, weight=weight2, max_new_tokens=100, output_scores=True, return_dict_in_generate=True)
                            gen = self.processor.decode(output[0][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True, output_attentions=True)

                        else:
                            output = self.model.generate(**single_input, keys=keys, weight=weight, max_new_tokens=100, output_scores=True, return_dict_in_generate=True)
                            uncertainty = np.round(float(max(torch.nn.functional.softmax(output['scores'][0], dim=-1)[0])), 2)
                            gen = self.processor.decode(output[0][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True, output_attentions=True)
                        
                        # Check correctness of the generated response
                        label = int(batch['labels'][0][idx])
                        if label == 1:
                            TP += 1 if 'Yes' in gen else 0
                            FN += 1 if 'Yes' not in gen else 0
                        else:
                            TN += 1 if 'No' in gen else 0
                            FP += 1 if 'No' not in gen else 0
                        
                        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
                        
                        # Create result entry for the current sample
                        gold = 'Yes' if label == 1 else 'No'
                        result = {
                            "Prompt": prompt,
                            "Generation": gen,
                            "Golden": gold,
                            "Uncertainty": uncertainty,
                        }
                        results.append(result)
                        index_of_total += 1
                        
                index += 1    
        # Calculate metrics
        precision = TP / (TP + FN)
        recall = TN / (TN + FP)
        f1_score = 2 * precision * recall / (precision + recall)

        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}\n"
            f"Accuracy: {(TN + TP) / (TN + TP + FN + FP)}\n"
            f"Precision: {precision}\n"
            f"Recall: {recall}\n"
            f"F1 Score: {f1_score}")
        
        all_scores = (TP, TN, FP, FN)
        
        # Save results to JSON file
        output_file_path = f'./output/results_{dataset}_{method}_{weight}.json'
        with open(output_file_path, 'w', encoding='utf-8') as fout:
            json.dump(results, fout, ensure_ascii=False, indent=4)
        
        # Save evaluation metrics
        output_score_file = output_file_path.replace(".json", "_scores.json")
        with open(output_score_file, 'w', encoding='utf-8') as fout:
            json.dump({"acc": (TN + TP) / (TN + TP + FN + FP), "precision": precision, "recall": recall, "f1": f1_score}, fout, ensure_ascii=False, indent=4)
        return all_scores
    
    @torch.no_grad()
    def bidirectional_reasoning_single(self, image, question, option_list):
        reasoning_chain = []

        default_fewshots = ''
        # Question 정제 및 객체 추출
        is_formatted_prompt = '<image>' in question or 'USER:' in question
        if is_formatted_prompt:
            question = question.replace('<image>', '').replace('USER:', '').replace('ASSISTANT:', '').strip()
        
        obj1, obj2 = self._extract_objects_from_question(question)
        
        if obj1 is None or obj2 is None:
            return self._direct_query_fallback(image, question, option_list, reasoning_chain)
        
        reasoning_chain.append({
            'step': 1,
            'action': 'extract_objects',
            'obj1': obj1,
            'obj2': obj2
        })
        
        # ============================================
        # Step 2: Direct Bidirectional Queries (no abstraction!)
        # ============================================
        
        # Step 2a: Forward
        query_step2a = f"Where is the {obj1} in relation to the {obj2}? Answer with left, right, on or under"
        prompt_step2a = f"<image>\nUSER: {query_step2a}\nASSISTANT:"
        
        inputs = self.processor(images=image, text=prompt_step2a, return_tensors="pt").to(self.device, torch.float16)
        output = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        response_forward = self.processor.decode(
            output.sequences[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        probs_forward = [torch.log_softmax(s[0], dim=-1).max().item() for s in output.scores]
        
        reasoning_chain.append({
            'step': '2a',
            'query': query_step2a,
            'response': response_forward,
            'avg_prob': float(np.mean(probs_forward))
        })
        
        # Step 2b: Reverse
        query_step2b = f"Where is the {obj2} in relation to the {obj1}? Answer with left, right, on or under"
        prompt_step2b = f"<image>\nUSER: {query_step2b}\nASSISTANT:"
        
        inputs = self.processor(images=image, text=prompt_step2b, return_tensors="pt").to(self.device, torch.float16)
        output = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        response_reverse = self.processor.decode(
            output.sequences[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        probs_reverse = [torch.log_softmax(s[0], dim=-1).max().item() for s in output.scores]
        
        reasoning_chain.append({
            'step': '2b',
            'query': query_step2b,
            'response': response_reverse,
            'avg_prob': float(np.mean(probs_reverse))
        })
        
        # ============================================
        # Step 3: Consistency Check
        # ============================================
        is_consistent = check_logical_consistency(response_forward, response_reverse)
        
        if is_consistent:
            # Consistency가 보장되면 forward response 사용
            final_prediction = response_forward
            avg_prob = float(np.mean(probs_forward))
            
            reasoning_chain.append({
                'step': 3,
                'action': 'consistency_verified',
                'is_consistent': True,
                'final_answer': final_prediction
            })
            
            return {
                'final_prediction': final_prediction,
                'is_consistent': True,
                'reasoning_chain': reasoning_chain,
                'avg_prob': avg_prob
            }
        else:
            # Inconsistent: fallback
            reasoning_chain.append({
                'step': 3,
                'action': 'inconsistency_detected',
                'is_consistent': False,
                'forward': response_forward,
                'reverse': response_reverse
            })
            
            return self._direct_query_fallback(image, question, option_list, reasoning_chain, inconsistent=True)

    def bidirectional_reasoning_2(self, image, question, option_list):
        reasoning_chain = []

        default_fewshots = ''
        # Question 정제 및 객체 추출
        is_formatted_prompt = '<image>' in question or 'USER:' in question
        if is_formatted_prompt:
            question = question.replace('<image>', '').replace('USER:', '').replace('ASSISTANT:', '').strip()
        
        obj1, obj2 = self._extract_objects_from_question(question)
        
        if obj1 is None or obj2 is None:
            return self._direct_query_fallback(image, question, option_list, reasoning_chain)
        
        reasoning_chain.append({
            'step': 1,
            'action': 'extract_objects',
            'obj1': obj1,
            'obj2': obj2
        })
        
        # ============================================
        # Step 1
        # ============================================
        query_step1 = f"""Looking at the {obj1} and the {obj2} in the image, what type of spatial relationship exists between them?
        Choose one of the following:
            - 'left/right relationship' (if they are side by side horizontally)
            - 'on/under relationship' (if one is above or below the other)

        Answer with just one of these two options."""
        prompt_step1 = f"<image>\nUSER: {query_step1}\nASSISTANT:"
        inputs = self.processor(images=image, text=prompt_step1, return_tensors="pt").to(self.device, torch.float16)
        output = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        response_step1 = self.processor.decode(
            output.sequences[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        reasoning_chain.append({
            'step': '1',
            'query': query_step1,
            'response': response_step1,
        })
        # ============================================
        # Step 2: Direct Bidirectional Queries (no abstraction!)
        # ============================================
        if 'left' in response_step1.lower() or 'right' in response_step1.lower():
            relation_type = 'left/right'
        elif 'on' in response_step1.lower() or 'under' in response_step1.lower():
            relation_type = 'on/under'
        else:
            print(reasoning_chain)
            return self._direct_query_fallback(image, question, option_list, reasoning_chain, inconsistent=True)
        
        relation_type = 'left/right' if 'left' in response_step1.lower() or 'right' in response_step1.lower() else 'on/under'
        options_map = {
            'left/right': "'left' or 'right'",
            'on/under': "'on' or 'under'",
        }
        options = options_map[relation_type]
        # Step 2a: Forward
        query_step2a = f"Where is the {obj1} in relation to the {obj2}? Answer in a word with {options}"
        prompt_step2a = f"<image>\nUSER: {query_step2a}\nASSISTANT:"
        
        inputs = self.processor(images=image, text=prompt_step2a, return_tensors="pt").to(self.device, torch.float16)
        output = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        response_forward = self.processor.decode(
            output.sequences[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        probs_forward = [torch.log_softmax(s[0], dim=-1).max().item() for s in output.scores]
        
        reasoning_chain.append({
            'step': '2a',
            'query': query_step2a,
            'response': response_forward,
            'avg_prob': float(np.mean(probs_forward))
        })
        
        # Step 2b: Reverse
        query_step2b = f"Where is the {obj2} in relation to the {obj1}? Answer in a word with {options}"
        prompt_step2b = f"<image>\nUSER: {query_step2b}\nASSISTANT:"
        
        inputs = self.processor(images=image, text=prompt_step2b, return_tensors="pt").to(self.device, torch.float16)
        output = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        response_reverse = self.processor.decode(
            output.sequences[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        probs_reverse = [torch.log_softmax(s[0], dim=-1).max().item() for s in output.scores]
        
        reasoning_chain.append({
            'step': '2b',
            'query': query_step2b,
            'response': response_reverse,
            'avg_prob': float(np.mean(probs_reverse))
        })
        
        # ============================================
        # Step 3: Consistency Check
        # ============================================
        is_consistent = check_logical_consistency(response_forward, response_reverse)
        
        if is_consistent:
            # Consistency가 보장되면 forward response 사용
            final_prediction = response_forward
            avg_prob = float(np.mean(probs_forward))
            
            reasoning_chain.append({
                'step': 3,
                'action': 'consistency_verified',
                'is_consistent': True,
                'final_answer': final_prediction
            })
            
            print(reasoning_chain)
            return {
                'final_prediction': final_prediction,
                'is_consistent': True,
                'reasoning_chain': reasoning_chain,
                'avg_prob': avg_prob
            }
        else:
            # Inconsistent: fallback
            reasoning_chain.append({
                'step': 3,
                'action': 'inconsistency_detected',
                'is_consistent': False,
                'forward': response_forward,
                'reverse': response_reverse
            })
            
            print(reasoning_chain)
            return self._direct_query_fallback(image, question, option_list, reasoning_chain, inconsistent=True)

    def _extract_objects_from_question(self, question):
        """
        질문에서 객체 이름 추출
        예: "Where is the beer bottle in relation to the armchair?" 
        -> ("beer bottle", "armchair")
        """
        import re
        
        # "Where is the X in relation to the Y?" 패턴
        pattern1 = r"where is (?:the )?(.+?) in relation to (?:the )?(.+?)\?"
        match = re.search(pattern1, question.lower())
        
        if match:
            obj1 = match.group(1).strip()  # subject
            obj2 = match.group(2).strip()  # reference
            return obj1, obj2
        
        # "What is the relationship between X and Y?" 패턴
        pattern2 = r"where are (?:the )?(.+?) in relation to (?:the )?(.+?)\?"
        match = re.search(pattern2, question.lower())
        
        if match:
            obj1 = match.group(1).strip()
            obj2 = match.group(2).strip()
            return obj1, obj2
        
        
        return None, None

    def _direct_query_fallback(self, image, question, option_list, reasoning_chain, inconsistent=False):
        """Direct query fallback"""
        if '<image>' in question or 'USER:' in question:
            question = question.replace('<image>', '').replace('USER:', '').replace('ASSISTANT:', '').strip()
        
        prompt = f"<image>\nUSER: {question}\nASSISTANT:"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        
        output = self.model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        final_prediction = self.processor.decode(
            output.sequences[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        probs = [torch.softmax(s[0], dim=-1).max().item() for s in output.scores]
        
        reasoning_chain.append({
            'step': 'fallback',
            'query': question,
            'response': final_prediction,
            'avg_prob': float(np.mean(probs)),
            'reason': 'inconsistent' if inconsistent else 'parsing_failed'
        })
        
        return {
            'final_prediction': final_prediction,
            'is_consistent': False,
            'reasoning_chain': reasoning_chain,
            'avg_prob': float(np.mean(probs)),
            'fallback': True
        }

    def _catch_relation(self, answer):
        answer = answer.lower()
        RELATIONS = ["right", "left", "on", "under"]
        candidates = []
        for r in RELATIONS:
            if r in answer:
                candidates.append(r)
        return candidates if len(candidates) > 0 else [answer]
    
    @torch.no_grad()
    def adaptvis_bidirectional_reasoning(self, image, question, option_list, threshold, weight1, weight2, sample_idx=None):
        """
        AdaptVis + Bidirectional Selection with file-based debugging
        """
        reasoning_chain = []
        
        question = question.replace('<image>', '').replace('USER:', '').replace('ASSISTANT:', '').strip()
        
        # 객체 추출
        obj1, obj2 = self._extract_objects_from_question(question)
        
        if obj1 is None or obj2 is None:
            result = self._direct_query_fallback(image, question, option_list, reasoning_chain)
            self._save_debug_log(reasoning_chain, sample_idx)
            return result
        
        
        reasoning_chain.append({
            'step': 1,
            'action': 'extract_objects',
            'obj1': obj1,
            'obj2': obj2
        })
        
        # ============================================
        # Step 2: Uncertainty 측정 (AdaptVis)
        # ============================================
        query_uncertainty = f"Where is the {obj1} in relation to the {obj2}? Answer with left, right, on or under"
        prompt_uncertainty = f"USER: <image>\n{query_uncertainty} ASSISTANT:"
        
        single_input = self.processor(
            text=prompt_uncertainty, 
            images=image, 
            padding="max_length", 
            return_tensors="pt", 
            max_length=77
        ).to(self.device)
        
        # Keys 생성
        keys_base = [torch.where(input_id == 32001, 1, 0) for input_id in single_input['input_ids']]
        change_greedy_to_add_weight()
        output_base = self.model.generate(
            **single_input,
            keys=keys_base,
            weight=1.0,
            max_new_tokens=100, 
            output_scores=True, 
            return_dict_in_generate=True
        )
        
        baseline_response = self.processor.decode(
            output_base['sequences'][0][len(single_input['input_ids'][-1]):], 
            skip_special_tokens=True
        ).strip()
        baseline_response_rs = self._catch_relation(baseline_response)
        baseline_response_r = baseline_response_rs[-1]

        # Uncertainty 계산
        uncertainty = np.round(float(max(torch.nn.functional.softmax(output_base['scores'][0], dim=-1)[0])), 2)
        
        # Threshold 기반 weight 선택
        if uncertainty < threshold:
            selected_weight = weight1
            weight_reason = 'low_confidence'
        else:
            selected_weight = weight2
            weight_reason = 'high_confidence'
        
        reasoning_chain.append({
            'step': 2,
            'action': 'adaptvis_weight_selection',
            'uncertainty': uncertainty,
            'threshold': threshold,
            'selected_weight': selected_weight,
            'reason': weight_reason,
            'baseline_response': baseline_response,
            'baseline_response_r': baseline_response_r,
            'num_catched_relation': len(baseline_response_rs)
        })

        # ============================================
        # Step 3a: Forward Query with Selected Weight
        # ============================================
        query_forward = f"Where is the {obj1} in relation to the {obj2}? Answer with left, right, on or under"
        prompt_forward = f"USER: <image>\n{query_forward} ASSISTANT:"
        
        single_input_forward = self.processor(
            text=prompt_forward, 
            images=image, 
            padding="max_length", 
            return_tensors="pt", 
            max_length=77
        ).to(self.device)
        
        keys_forward = [torch.where(input_id == 32001, 1, 0) for input_id in single_input_forward['input_ids']]
        
        output_forward = self.model.generate(
            **single_input_forward, 
            keys=keys_forward, 
            weight=selected_weight,
            max_new_tokens=100, 
            output_scores=True, 
            return_dict_in_generate=True
        )
        
        response_forward = self.processor.decode(
            output_forward['sequences'][0][len(single_input_forward['input_ids'][-1]):], 
            skip_special_tokens=True
        ).strip()
        response_forward_rs = self._catch_relation(response_forward)
        response_forward_r = response_forward_rs[-1]

        # Forward confidence 계산
        probs_forward = [torch.softmax(s[0], dim=-1).max().item() for s in output_forward['scores']]
        confidence_forward = float(np.mean(probs_forward))
        
        reasoning_chain.append({
            'step': '3a',
            'direction': 'forward',
            'query': query_forward,
            'response': response_forward,
            'response_r': response_forward_r,
            'num_catched_relation': len(response_forward_rs),
            'weight': selected_weight,
            'confidence': confidence_forward,
            'token_probs': [float(p) for p in probs_forward]
        })
        
        # ============================================
        # Step 3b: Reverse Query with Same Weight
        # ============================================
        query_reverse = f"Where is the {obj2} in relation to the {obj1}? Answer with left, right, on or under"
        prompt_reverse = f"USER: <image>\n{query_reverse} ASSISTANT:"
        
        single_input_reverse = self.processor(
            text=prompt_reverse, 
            images=image, 
            padding="max_length", 
            return_tensors="pt", 
            max_length=77
        ).to(self.device)
        
        keys_reverse = [torch.where(input_id == 32001, 1, 0) for input_id in single_input_reverse['input_ids']]
        
        output_reverse = self.model.generate(
            **single_input_reverse, 
            keys=keys_reverse, 
            weight=selected_weight,
            max_new_tokens=100, 
            output_scores=True, 
            return_dict_in_generate=True
        )
        
        response_reverse = self.processor.decode(
            output_reverse['sequences'][0][len(single_input_reverse['input_ids'][-1]):], 
            skip_special_tokens=True
        ).strip()
        response_reverse_rs = self._catch_relation(response_reverse)
        response_reverse_r = response_reverse_rs[-1]

        # Reverse confidence 계산
        probs_reverse = [torch.softmax(s[0], dim=-1).max().item() for s in output_reverse['scores']]
        confidence_reverse = float(np.mean(probs_reverse))
        
        
        reasoning_chain.append({
            'step': '3b',
            'direction': 'reverse',
            'query': query_reverse,
            'response': response_reverse,
            'response_r': response_reverse_r,
            'num_catched_relation': len(response_reverse_rs),
            'weight': selected_weight,
            'confidence': confidence_reverse,
            'token_probs': [float(p) for p in probs_reverse]
        })
        
        # ============================================
        # Step 4: Bidirectional Selection
        # ============================================
        
        # logic 추가
        if len(response_forward_rs) == len(response_reverse_rs) == 1:
            # Consistency 검증
            is_consistent = check_logical_consistency(response_forward_r, response_reverse_r)
            expected_reverse = get_symmetric_relation(response_forward_r.lower())
            
            
            # Confidence 차이
            confidence_diff = abs(confidence_forward - confidence_reverse)
            
            
            if is_consistent:
                # 일관성 있음
                final_prediction = response_forward_r
                selected_confidence = confidence_forward
                selection_reason = 'consistent_forward'
                selected_direction = 'forward'
                
                
            else:
                # 일관성 없음
                
                if confidence_forward >= confidence_reverse:
                    final_prediction = response_forward_r
                    selected_confidence = confidence_forward
                    selection_reason = 'higher_confidence_forward'
                    selected_direction = 'forward'
                    
                else:
                    # Reverse가 더 확신함 → 대칭 변환
                    final_prediction = get_symmetric_relation(response_reverse_r.lower())
                    selected_confidence = confidence_reverse
                    selection_reason = 'higher_confidence_reverse_symmetric'
                    selected_direction = 'reverse'
                
        
            reasoning_chain.append({
                'step': 4,
                'action': 'bidirectional_selection',
                'is_consistent': is_consistent,
                'confidence_forward': confidence_forward,
                'confidence_reverse': confidence_reverse,
                'confidence_diff': confidence_diff,
                'response_forward': response_forward,
                'response_reverse': response_reverse,
                'expected_reverse': expected_reverse,
                'selected_direction': selected_direction,
                'final_answer': final_prediction,
                'selection_reason': selection_reason
            })
        
        else:
            if confidence_forward >= confidence_reverse:
                query_final = f"Where is the {obj1} in relation to the {obj2}? Answer in a word left, right, on or under, with reference to the following observation: {response_forward}"
                prompt_final = f"USER: <image>\n{query_final} ASSISTANT:"
                
                single_input = self.processor(
                    text=prompt_final, 
                    images=image, 
                    padding="max_length", 
                    return_tensors="pt", 
                    max_length=77
                ).to(self.device)
                
                # Keys 생성
                keys_final = [torch.where(input_id == 32001, 1, 0) for input_id in single_input['input_ids']]
                change_greedy_to_add_weight()
                output_final = self.model.generate(
                    **single_input,
                    keys=keys_final,
                    weight=1.0,
                    max_new_tokens=100, 
                    output_scores=True, 
                    return_dict_in_generate=True
                )
                
                final_response = self.processor.decode(
                    output_final['sequences'][0][len(single_input['input_ids'][-1]):], 
                    skip_special_tokens=True
                ).strip()
                final_response_rs = self._catch_relation(final_response)
                final_response_r = final_response_rs[-1]

            reasoning_chain.append({
                'step': 4,
                'action': 'bidirectional_selection',
                'is_consistent': None,
                'confidence_forward': confidence_forward,
                'confidence_reverse': confidence_reverse,
                'confidence_diff': confidence_diff,
                'response_forward': response_forward,
                'response_reverse': response_reverse,
                'expected_reverse': None,
                'selected_direction': None,
                'final_answer': final_response_r,
                'selection_reason': None
            })

        
        # 디버깅 정보 저장
        self._save_debug_log(reasoning_chain, sample_idx, {
            'question': question,
            'obj1': obj1,
            'obj2': obj2,
            'uncertainty': uncertainty,
            'selected_weight': selected_weight,
            'response_forward': response_forward,
            'response_reverse': response_reverse,
            'confidence_forward': confidence_forward,
            'confidence_reverse': confidence_reverse,
            'is_consistent': is_consistent,
            'final_prediction': final_prediction
        })
        
        return {
            'final_prediction': final_prediction,
            'is_consistent': is_consistent,
            'confidence': selected_confidence,
            'confidence_forward': confidence_forward,
            'confidence_reverse': confidence_reverse,
            'selected_weight': selected_weight,
            'initial_uncertainty': uncertainty,
            'selection_reason': selection_reason,
            'reasoning_chain': reasoning_chain,
            'avg_prob': selected_confidence
        }

    def _save_debug_log(self, reasoning_chain, sample_idx, summary=None):
        """디버깅 정보를 파일로 저장"""
        import os
        import json
        from datetime import datetime
        
        # temp 디렉토리 생성
        os.makedirs('./temp1', exist_ok=True)
        
        # 파일명 생성 (sample_idx가 있으면 사용, 없으면 타임스탬프)
        if sample_idx is not None:
            filename = f'sample_{sample_idx:05d}.json'
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f'sample_{timestamp}.json'
        
        filepath = os.path.join('./temp', filename)
        
        # 저장할 데이터 구성
        debug_data = {
            'sample_idx': sample_idx,
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'reasoning_chain': reasoning_chain
        }
        
        # JSON으로 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, ensure_ascii=False, indent=2)



