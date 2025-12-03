import torch
import numpy as np
from tqdm import tqdm
import random
from transformers import AutoProcessor, LlamaTokenizerFast, CLIPImageProcessor
from .llava import  LlavaForConditionalGeneration, LlavaForConditionalGenerationScal

import torch
import torch.nn.functional as F
import json
import os
import re

import warnings
from typing import List, Optional, Union

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import GenerateEncoderDecoderOutput,GenerateDecoderOnlyOutput,GenerateNonBeamOutput
import os
import json
import random
import numpy as np
import torch
from tqdm import tqdm
MODEL='llava-hf/llava-1.5-7b-hf'

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

class LlavaWrapper:
    def __init__(self, root_dir, device,method):
        
        if method=='scaling_vis' or ('adapt_vis' in method) or method=='adaptvis_bidirectional':
            self.model = LlavaForConditionalGenerationScal.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir,ignore_mismatched_sizes=True).eval().to(device)

        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(MODEL, revision='a272c74', cache_dir=root_dir,ignore_mismatched_sizes=True).eval().to(device)

        self.feature_extractor = CLIPImageProcessor.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir)
        self.tokenizer = LlamaTokenizerFast.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir)
        self.processor = AutoProcessor.from_pretrained(MODEL, revision='a272c74',cache_dir=root_dir)

        self.device = device
    
    def _kl(self, p, q):
        return torch.sum(p * torch.log(p / q))

    def _jsd(self, p, q):
        # 1. 평균 분포(Mean Distribution) M 계산
        m = 0.5 * (p + q)
        
        # 2. 각 KL 항 계산
        kl_pm = torch.sum(p * torch.log(p / m)) # P와 M 사이의 KL
        kl_qm = torch.sum(q * torch.log(q / m)) # Q와 M 사이의 KL
        
        # 3. JSD 반환 (결과는 0 ~ ln(2) 사이)
        return 0.5 * (kl_pm + kl_qm)
    
    def get_distribution(self, score, dataset="Controlled_Images_A"):
        # 1. Dataset에 따른 Option 설정 (기존 로직 동일)
        if dataset == "Controlled_Images_A":
            options = ["Left", "Right", "On", "Under"]
        elif dataset == "Controlled_Images_B":
            options = ["Left", "Right", "Front", "Behind"]
        elif dataset == "COCO_QA_one_obj":
            options = ["Left", "Right", "Top", "Bottom"]
        elif dataset == "COCO_QA_two_obj":
            options = ["Left", "Right", "Above", "Below"]
        elif dataset == "VG_QA_one_obj":
            options = ["Left", "Right", "Front", "Behind", "Top", "Bottom"]
        elif dataset == "VG_QA_two_obj":
            options = ["Left", "Right", "Front", "Behind", "Above", "Below"]

        # 2. Option에 대한 확률 추출 및 정규화
        option_ids = [self.tokenizer.encode(r, add_special_tokens=False)[0] for r in options]
        prob_options = score[0, option_ids]
        
        # Softmax를 통해 확률 분포로 변환 (합이 1이 되도록)
        p = F.softmax(prob_options, dim=-1)
        
        # 수치 안정성 (log 계산 시 NaN 방지)
        p = p + 1e-12
        p = p / p.sum() 
        
        N = len(options) # 선택지 개수
        return {
            options[i]: float(p[i].item()) for i in range(N)
        }
        
    def get_uncertainty(self, score, distribution, method='entropy'):
        N = len(distribution)
        p = torch.Tensor([
            distribution[key] for key in distribution
        ]).to(score.device)
        
        # 3. Uncertainty 측정 방식 선택
        if method == 'confidence':
            res = float(max(torch.nn.functional.softmax(score, dim=-1)[0]))
        
        elif method == 'kld':
            uniform_dist = torch.ones_like(p) / N
            res = float(self._kl(p, uniform_dist).item())
        
        elif method == 'jsd':
            uniform_dist = torch.ones_like(p) / N
            res = float(self._jsd(p, uniform_dist).item())
        
        elif method == 'entropy':
            # (1) Entropy 계산: -sum(p * log(p))
            entropy = -torch.sum(p * torch.log(p))
            
            # (2) Normalization: Divide by log(N)
            # 결과는 0 (확신) ~ 1 (완전 모름) 사이
            max_entropy = torch.log(torch.tensor(float(N)))
            normalized_entropy = entropy / max_entropy
            
            res = float(normalized_entropy.item())
        
        else:
            res = float(max(torch.nn.functional.softmax(score, dim=-1)[0]))
            
        return res
    
    def get_confidence_sentence(self, score):
        return float(np.exp((np.mean([torch.log_softmax(s[0], dim=-1).max().item() for s in score]))))
    
    def get_confidence_where(self, score, idx):
        s = score[idx]
        return float(torch.softmax(s[0], dim=-1).max().item())
        
    @torch.no_grad()
    def get_answer(self, prompt, image, weight=None, max_length=77, max_new_tokens=100, get_token_probs=False):
        single_input = self.processor(
            text=prompt, images=image, padding="max_length", return_tensors="pt", max_length=max_length
        ).to(self.device)
        
        if weight is None:
            output = self.model.generate(
                **single_input, max_new_tokens=max_new_tokens, output_scores=True, return_dict_in_generate=True
            )
        else:
            keys = [torch.where(input_id == 32001, 1, 0) for input_id in single_input['input_ids']]
            output = self.model.generate(
                **single_input, keys=keys, weight=weight, max_new_tokens=max_new_tokens, output_scores=True, return_dict_in_generate=True
            )
        
        gen = self.processor.decode(output['sequences'][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True)
        scores = output['scores']
        if not get_token_probs:
            return gen, scores
        else:
            transition_scores = self.model.compute_transition_scores(
                output.sequences, 
                output.scores, 
                normalize_logits=True
            )
            input_length = single_input['input_ids'].shape[1]
            generated_tokens = output.sequences[:, input_length:]

            # 토큰과 확률 짝짓기
            token_probs = []
            for tok, score in zip(generated_tokens[0], transition_scores[0]):
                token_id = tok.item()
                token = self.processor.decode(token_id)
                probability = torch.exp(score).item()  # log prob -> prob
                token_probs.append((token_id, token, probability))
                
            return gen, scores, token_probs
    
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
        save_attn_dir_weight = f"./output/{dataset}_method_{method}_weight_{weight:.2f}"
        os.makedirs(save_attn_dir_weight, exist_ok=True)

        results = []  # Store results for each generated sequence
        output_result_file_path = None
        for batch in tqdm(joint_loader):
            batch_scores = []
            
            # Set environment variable for attention map save path
            os.environ['SAVE_ATTN_PATH'] = f'{save_attn_dir_weight}/{index_of_total}/'
            os.makedirs(os.environ['SAVE_ATTN_PATH'], exist_ok=True)
            
            for i_option in batch["image_options"]:
                im_scores = []
                for _ in i_option:
                    result = None
                    prompt = prompt_list[index_of_total]
                    
                    if method=='reasoning_1':
                        def consistency_check(gen1, gen2):
                            valid_opposite = {
                                'left': 'right',
                                'right': 'left',
                                'top': 'bottom',
                                'bottom': 'top',
                            }
                            gen1 = gen1.lower()
                            gen2 = gen2.lower()
                            
                            if gen1 in valid_opposite and gen2 == valid_opposite[gen1]:
                                return True
                            else:
                                return False
                        
                        # Step 1: 객체 추출 및 두 객체의 이미지상 절대적 위치 파악
                        pattern = r"Where (is|are) the (.+?) in relation to the (.+?)\?"
                        match = re.search(pattern, prompt)
                        be_verb, obj1, obj2 = match.group(1), match.group(2), match.group(3)
                        
                        prompt_step1_obj1 = f"<image>\nUSER: Where {be_verb} the {obj1} located in the image? Answer with left, right, top or bottom.\nASSISTANT:"
                        prompt_step1_obj2 = f"<image>\nUSER: Where is the {obj2} located in the image? Answer with left, right, top or bottom.\nASSISTANT:"
                        gen_step1_obj1, l = self.get_answer(prompt_step1_obj1, _)
                        gen_step1_obj2, l = self.get_answer(prompt_step1_obj2, _)
                        
                        # Step 2: Consistency 체크
                        consistent = consistency_check(gen_step1_obj1, gen_step1_obj2)
                        
                        # Step 3: 최종 답변
                        if consistent:
                            prompt = f"<image>\nUSER: The {obj1} {be_verb} positioned {gen_step1_obj1}-side on the image, and the {obj2} is positioned {gen_step1_obj2}-side on the image. Then, Where {be_verb} the {obj1} in relation to the {obj2}? Answer about the relation between the {obj1} and the {obj2} with left, right, on or under.\nASSISTANT: "
                            gen, score = self.get_answer(prompt, _)
                            score = score[0]
                            uncertainty = np.round(float(max(torch.nn.functional.softmax(score, dim=-1)[0])), 2)
                            
                        else:
                            gen, score = self.get_answer(prompt, _)
                            score = score[0]
                            uncertainty = np.round(float(max(torch.nn.functional.softmax(score, dim=-1)[0])), 2)

                        result = {
                            "Prompt": prompt,
                            "Step1_1_to_2": gen_step1_obj1,
                            "Step1_2_to_1": gen_step1_obj2,
                            "Is_consistent": consistent,
                            "Generation": gen,
                            "Golden": answer_list[index_of_total][0],
                            "uncertainty": uncertainty
                        }
                        
                    elif method=='reasoning_2':
                        def consistency_check(gen1, gen2):
                            valid_opposite = {
                                'left': 'right',
                                'right': 'left',
                                'on': 'under',
                                'under': 'on',
                            }
                            gen1 = gen1.lower()
                            gen2 = gen2.lower()
                            
                            if gen1 in valid_opposite and gen2 == valid_opposite[gen1]:
                                return True
                            else:
                                return False
                        
                        # Step 1: 객체 추출 및 두 객체의 이미지상 상대적 위치 파악
                        pattern = r"Where (is|are) the (.+?) in relation to the (.+?)\?"
                        match = re.search(pattern, prompt)
                        be_verb, obj1, obj2 = match.group(1), match.group(2), match.group(3)
                        
                        prompt_step1_obj1 = f"<image>\nUSER: Where {be_verb} the {obj1} in relation to the {obj2}? Answer with left, right, on or under.\nASSISTANT:"
                        prompt_step1_obj2 = f"<image>\nUSER: Where is the {obj2} in relation to the {obj1}? Answer with left, right, on or under.\nASSISTANT:"
                        gen_step1_obj1, score_step1_obj1 = self.get_answer(prompt_step1_obj1, _)
                        score_step1_obj1 = score_step1_obj1[0]
                        gen_step1_obj2, score_step1_obj2 = self.get_answer(prompt_step1_obj2, _)
                        
                        # Step 2: Consistency 체크
                        consistent = consistency_check(gen_step1_obj1, gen_step1_obj2)
                        
                        # Step 3: 최종 답변
                        if consistent:
                            gen, score = gen_step1_obj1, score_step1_obj1
                            uncertainty = np.round(float(max(torch.nn.functional.softmax(score, dim=-1)[0])), 2)
                        else:
                            gen, score = self.get_answer(prompt, _)
                            score = score[0]
                            uncertainty = np.round(float(max(torch.nn.functional.softmax(score, dim=-1)[0])), 2)
                        
                        result = {
                            "Prompt": prompt,
                            "Step1_1_to_2": gen_step1_obj1,
                            "Step1_2_to_1": gen_step1_obj2,
                            "Is_consistent": consistent,
                            "Generation": gen,
                            "Golden": answer_list[index_of_total][0],
                            "uncertainty": uncertainty
                        }
                       
                    elif method=='reasoning_3':
                        # Step 1: 객체 추출 및 두 객체의 위치 관계 파악
                        pattern = r"Where (is|are) the (.+?) in relation to the (.+?)\?"
                        match = re.search(pattern, prompt)
                        be_verb, obj1, obj2 = match.group(1), match.group(2), match.group(3)
                        
                        prompt_step1 = f"<image>\nUSER: Which of the following positional relationships do the {obj1} and the {obj2} have? 1. A left-right relationship in which one object is next to another or 2. an on-under relationship in which one object is placed on or under another object.\nASSISTANT:"
                        gen_step1, score_step1 = self.get_answer(prompt_step1, _)
                        
                        # Step 2: 최종 답변
                        if 'left' in gen_step1 or 'right' in gen_step1 or '1' in gen_step1:
                            prompt_step2 = f"<image>\nUSER: Where {be_verb} the {obj1} in relation to the {obj2}? Answer with left or right.\nASSISTANT:"
                            gen, score = self.get_answer(prompt_step2, _)
                            score = score[0]
                            uncertainty = np.round(float(max(torch.nn.functional.softmax(score, dim=-1)[0])), 2)
                        elif 'on' in gen_step1 or 'under' in gen_step1 or '2' in gen_step1:
                            prompt_step2 = f"<image>\nUSER: Where {be_verb} the {obj1} in relation to the {obj2}? Answer with on or under.\nASSISTANT:"
                            gen, score = self.get_answer(prompt_step2, _)
                            score = score[0]
                            uncertainty = np.round(float(max(torch.nn.functional.softmax(score, dim=-1)[0])), 2)
                        else:
                            gen, score = self.get_answer(prompt, _)
                            score = score[0]
                            uncertainty = np.round(float(max(torch.nn.functional.softmax(score, dim=-1)[0])), 2)
                        
                        result = {
                            "Prompt": prompt,
                            "Step1": gen_step1,
                            "Generation": gen,
                            "Golden": answer_list[index_of_total][0],
                            "uncertainty": uncertainty
                        }
                    
                    elif method == 'adapt_vis_var_uncertainties_var_weights':
                        '''change_greedy_to_add_weight()

                        output = self.model.generate(
                            **single_input,weight=1.0,max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                        )
                        first_token_scores = output['scores'][0]
                        uncertainty = np.round(float(max(torch.nn.functional.softmax(first_token_scores, dim=-1)[0])), 2)
                        uncertainty_confidence, _ = self.get_uncertainty(first_token_scores, method='confidence', dataset=dataset)
                        uncertainty_kl, _ = self.get_uncertainty(first_token_scores, method='kld', dataset=dataset)
                        uncertainty_js, _ = self.get_uncertainty(first_token_scores, method='jsd', dataset=dataset)
                        uncertainty_entropy, distribution_map = self.get_uncertainty(first_token_scores, method='entropy', dataset=dataset, get_distribution=True)
                        uncertainties = {
                            "confidence": uncertainty_confidence,
                            "kld": uncertainty_kl,
                            "jsd": uncertainty_js,
                            "entropy": uncertainty_entropy
                        }
                        original_gen = self.processor.decode(output['sequences'][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True)
                        
                        weights = [0.5, 0.8, 1.2, 1.5, 2.0]
                        gen_map = {
                            1.0: original_gen
                        }
                        for w in weights:
                            output = self.model.generate(
                                **single_input, keys=keys, weight=w, 
                                max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                            )
                            gen_w = self.processor.decode(output['sequences'][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True)
                            gen_map[w] = gen_w
                            
                        # Select final answer based on standard weights & threshold
                        gen1, gen2 = gen_map[weight1], gen_map[weight2]
                        gen = gen1 if uncertainty < threshold else gen2
                        output_result_file_path = f'./output/results_{dataset}_{method}_{weight}_{weight1}_{weight2}_{threshold}_{TEST}.json'

                        result = {
                            "Prompt": prompt,
                            "Generation": gen,
                            "Generation_map": gen_map,
                            "Distribution_map": distribution_map,
                            "Golden": answer_list[index_of_total][0],
                            "Uncertainty": uncertainty,
                            "Uncertainties": uncertainties
                        }
                        print(f"Prompt:\n{prompt}\nGeneration: {gen}\nGolden: {answer_list[index_of_total][0]}")'''
                        change_greedy_to_add_weight()
                        original_generation, original_score = self.get_answer(prompt, _, 1.0)
                        distribution_map = self.get_distribution(original_score, dataset=dataset)
                        uncertainty = np.round(float(max(torch.nn.functional.softmax(original_score, dim=-1)[0])), 2)
                        uncertainty_confidence = self.get_uncertainty(original_score, distribution_map, method='confidence')
                        uncertainty_kl = self.get_uncertainty(original_score, distribution_map, method='kld')
                        uncertainty_js = self.get_uncertainty(original_score, distribution_map, method='jsd')
                        uncertainty_entropy = self.get_uncertainty(original_score, distribution_map, method='entropy')
                        uncertainties = {
                            "confidence": uncertainty_confidence,
                            "kld": uncertainty_kl,
                            "jsd": uncertainty_js,
                            "entropy": uncertainty_entropy
                        }
                        
                        weights = [0.5, 0.8, 1.2, 1.5, 2.0]
                        gen_map = {
                            1.0: {
                                "Generation": original_generation,
                                "Distribution": distribution_map
                            }
                        }
                        for w in weights:
                            w_generation, w_score = self.get_answer(prompt, _, w)
                            w_score = w_score[0]
                            distribution_map = self.get_distribution(w_score, dataset=dataset)
                            gen_map[w] = {
                                "Generation": w_generation,
                                "Distribution": distribution_map
                            }
                            
                        # Select final answer based on standard weights & threshold
                        gen1, gen2 = gen_map[weight1]['Generation'], gen_map[weight2]['Generation']
                        gen = gen1 if uncertainty < threshold else gen2
                        output_result_file_path = f'./output/results_{dataset}_{method}_{weight}_{weight1}_{weight2}_{threshold}_{TEST}.json'

                        result = {
                            "Prompt": prompt,
                            "Generation": gen,
                            "Generation_map": gen_map,
                            "Golden": answer_list[index_of_total][0],
                            "Uncertainty": uncertainty,
                            "Uncertainties": uncertainties
                        }
                    
                    elif method == 'few_shot_CoT':
                        few_shot_prompt = '''\
USER: Where is the violin in relation to the sofa? Think step by step, then answer about the relation between violin and sofa with left, right, on or under.
ASSISTANT: In this picture, the brown violin is lying on the floor, positioned beneath the beige fabric sofa. Since the instrument is located directly below the seat of the sofa, it is obscured by the sofa's shadow. So, as the final answer to the question of where the violin is in relation to the sofa, the violin is under the sofa.
USER: Where is the calculator in relation to the desk? Think step by step, then answer about the relation between calculator and desk with left, right, on or under.
ASSISTANT: In this picture, the gray calculator is placed on top of the wooden desk next to some papers. The device is resting on the desk's surface, clearly visible and accessible for use. So, as the final answer to the question of where the calculator is in relation to the desk, the calculator is on the desk.
USER: Where is the cat in relation to the rug? Think step by step, then answer about the relation between cat and rug with left, right, on or under.
ASSISTANT: In this picture, a white cat is sitting on the floor, adjacent to the patterned rug. Looking at the layout, the rug is in the center, and the cat is sitting towards the eastern side of the frame relative to the rug. So, as the final answer to the question of where the cat is in relation to the rug, the cat is right of the rug.
USER: Where is the stapler in relation to the printer? Think step by step, then answer about the relation between stapler and printer with left, right, on or under.
ASSISTANT: In this picture, the blue stapler is sitting on the desk surface, positioned to the side of the large laser printer. The printer occupies the right side of the view, while the stapler is placed on the opposite side. So, as the final answer to the question of where the stapler is in relation to the printer, the stapler is left of the printer.
'''
                        pattern = r"Where (is|are) the (.+?) in relation to the (.+?)\?"
                        match = re.search(pattern, prompt)
                        be_verb, obj1, obj2 = match.group(1), match.group(2), match.group(3)
                        new_prompt = f"<image>\nUSER: Where {be_verb} the {obj1} in relation to the {obj2}? Think step by step, then answer about the relation between the {obj1} and the {obj2} with left, right, on or under.\nASSISTANT:"
                        prompt = few_shot_prompt + new_prompt
                        
                        generation, score = self.get_answer(prompt, _, max_length=512, max_new_tokens=128)
                        answer = generation.split('.')[-2].strip()
                        print(f"Prompt:\n{new_prompt}\nGeneration: {answer}\nGolden: {answer_list[index_of_total][0]}")
                        
                        result = {
                            "Prompt": prompt,
                            "Generation": generation,
                            "Answer": answer,
                            "Golden": answer_list[index_of_total][0]
                        }
                       
                    elif method == 'few_shot_CoT_r1':
                        few_shot_prompt = '''\
USER: Where is the violin in relation to the sofa? Think step by step, then answer about the relation between violin and sofa with left, right, on or under.
ASSISTANT: In this picture, the brown violin is lying on the floor, positioned beneath the beige fabric sofa. Since the instrument is located directly below the seat of the sofa, it is obscured by the sofa's shadow. So, as the final answer to the question of where the violin is in relation to the sofa, the violin is under the sofa.
USER: Where is the calculator in relation to the desk? Think step by step, then answer about the relation between calculator and desk with left, right, on or under.
ASSISTANT: In this picture, the gray calculator is placed on top of the wooden desk next to some papers. The device is resting on the desk's surface, clearly visible and accessible for use. So, as the final answer to the question of where the calculator is in relation to the desk, the calculator is on the desk.
USER: Where is the cat in relation to the rug? Think step by step, then answer about the relation between cat and rug with left, right, on or under.
ASSISTANT: In this picture, a white cat is sitting on the floor, adjacent to the patterned rug. Looking at the layout, the rug is in the center, and the cat is sitting towards the eastern side of the frame relative to the rug. So, as the final answer to the question of where the cat is in relation to the rug, the cat is right of the rug.
USER: Where is the shoes in relation to the bed? Think step by step, then answer about the relation between shoes and bed with left, right, on or under.
ASSISTANT: In this picture, the black leather shoes are positioned on the floor, directly beneath the bed frame. They are located in the space between the floor and the mattress support. So, as the final answer to the question of where the shoes is in relation to the bed, the shoes are on the floor...Wait, The original question was to ask about the spatial relationship between the shoes and the bed, but I just ended up answering the relationship between the shoes and the floor. So, to correct the answer again, the shoes are under the bed.
USER: Where is the stapler in relation to the printer? Think step by step, then answer about the relation between stapler and printer with left, right, on or under.
ASSISTANT: In this picture, the blue stapler is sitting on the desk surface, positioned to the side of the large laser printer. The printer occupies the right side of the view, while the stapler is placed on the opposite side. So, as the final answer to the question of where the stapler is in relation to the printer, the stapler is left of the printer.
USER: Where is the lamp in relation to the monitor? Think step by step, then answer about the relation between lamp and monitor with left, right, on or under.
ASSISTANT: In this picture, the silver lamp is positioned on the desk surface, in front of the computer monitor. The monitor occupies the left side of the view, and the lamp is positioned on the right side. So, as the final answer to the question, the lamp is in front of the monitor... Wait, There's a contradiction in the reasoning I just made. I said the lamp is in front of the monitor and at the same time said the lamp is on the right side of the monitor. Hmm... Let's look at the image again. Aha! It was wrong to say that the lamp is in front of the monitor. So, to correct the answer again, the lamp is right of the monitor.
'''
                        pattern = r"Where (is|are) the (.+?) in relation to the (.+?)\?"
                        match = re.search(pattern, prompt)
                        be_verb, obj1, obj2 = match.group(1), match.group(2), match.group(3)
                        new_prompt = f"<image>\nUSER: Where {be_verb} the {obj1} in relation to the {obj2}? Think step by step, then answer about the relation between the {obj1} and the {obj2} with left, right, on or under.\nASSISTANT:"
                        prompt = few_shot_prompt + new_prompt
                        
                        generation, score, token_probs = self.get_answer(prompt, _, max_length=1024, max_new_tokens=128, get_token_probs=True)
                        answer = generation.split('.')[-2].strip()
                        print(f"Prompt:\n{new_prompt}\nGeneration: {answer}\nGolden: {answer_list[index_of_total][0]}")
                        
                        result = {
                            "Prompt": prompt,
                            "Generation": generation,
                            "Answer": answer,
                            "Golden": answer_list[index_of_total][0],
                            "token_probs": token_probs
                        }
                       
                     
                    else:
                        gen, score = self.get_answer(prompt, _)
                        uncertainty = np.round(float(max(torch.nn.functional.softmax(score, dim=-1)[0])), 2)
                        result = {
                            "Prompt": prompt,
                            "Generation": gen,
                            "Golden": answer_list[index_of_total][0],
                            "uncertainty": uncertainty
                        }
                        
                    result = {
                        "Prompt": prompt,
                        "Generation": gen,
                        "Golden": answer_list[index_of_total][0]
                    } if result is None else result
                    results.append(result)
                    
                    # Check if the generation matches the expected answer
                    c_option = batch["caption_options"]
                    if 'CoT' in method:
                        if len(list(c_option)) == 4:
                            answer = answer.lower()
                            if (answer_list[index_of_total][0].lower() in answer) and not ('floor' in answer):
                                acc += 1
                                correct_id.append(index_of_total)
                                answers = [1, 0, 0, 0]
                            
                            else:
                                answers = [0, 0, 1, 0]
                    else:
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
            output_result_file_path = f'./output/results_{dataset}_{method}_{weight}_{threshold}_{TEST}.json' if output_result_file_path is None else output_result_file_path
            with open(output_result_file_path, 'w', encoding='utf-8') as fout:
                json.dump(results, fout, ensure_ascii=False, indent=4)
            print(acc, index_of_total, acc / index_of_total)
                 
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
    
        
        
        
        
        
        