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
        p = p + 1e-12
        q = q + 1e-12
        p = p / p.sum()
        print(f"\nNormalized_probabilities: {p}")
        q = q / q.sum()
        return torch.sum(p * torch.log(p / q))

    def _jsd(self, p, q):
        """
        Jensen-Shannon Divergence 계산
        JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        where M = 0.5 * (P + Q)
        """
        # 1. 수치 안정성 및 정규화 (기존 로직 유지)
        # 부분집합(subset)의 확률 합이 1이 되도록 다시 맞춰줍니다.
        p = p + 1e-12
        q = q + 1e-12
        p = p / p.sum()
        q = q / q.sum()
        
        # 2. 평균 분포(Mean Distribution) M 계산
        m = 0.5 * (p + q)
        
        # 3. 각 KL 항 계산
        # P와 M 사이의 KL
        kl_pm = torch.sum(p * torch.log(p / m))
        # Q와 M 사이의 KL
        kl_qm = torch.sum(q * torch.log(q / m))
        
        # 4. JSD 반환 (결과는 0 ~ ln(2) 사이)
        return 0.5 * (kl_pm + kl_qm)
    
    def get_uncertainty(self, prob, method=None, dataset="Controlled_Images_A"):
        if method=='kl_divergence':
            if dataset == "Controlled_Images_A":
                options = ["Left", "Right", "On", "Under"]
            elif dataset == "Controlled_Images_B":
                options = ["Left", "Right", "Front", "Behind"]
            elif dataset == "VG_QA_two_obj":
                options = ["left", "right", "front", "behind", "above", "below"]
            option_ids = [self.tokenizer.encode(r, add_special_tokens=False)[0] for r in options]
            prob_options = prob[0, option_ids]
            prob_options = F.softmax(prob_options, dim=-1)
            print(f"\nProbabilities of options: {prob_options}")
            uniform_dist = torch.ones_like(prob_options) / len(option_ids)
            return float(self._jsd(prob_options, uniform_dist).item())
        else:
            return float(max(torch.nn.functional.softmax(prob, dim=-1)[0]))
    
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
        save_attn_dir_weight = f"./output/{dataset}_weight{weight:.2f}"
        save_attn_dir_weight1 = f"./output/{dataset}_weight{weight1:.2f}"
        save_attn_dir_weight2 = f"./output/{dataset}_weight{weight2:.2f}"
        os.makedirs(save_attn_dir_weight, exist_ok=True)
        os.makedirs(save_attn_dir_weight1, exist_ok=True)
        os.makedirs(save_attn_dir_weight2, exist_ok=True)

        results = []  # Store results for each generated sequence
        reasoning_chains = []
        for batch in tqdm(joint_loader):
            batch_scores = []
            
            # Set environment variable for attention map save path
            os.environ['SAVE_ATTN_PATH'] = f'{save_attn_dir_weight}/{index_of_total}/'
            os.makedirs(os.environ['SAVE_ATTN_PATH'], exist_ok=True)

            # Iterate over each image option in the batch
            for i_option in batch["image_options"]:
                im_scores = []
                uncertainty_prob = None
                uncertainty_kl = None
                uncertainty = None
                for _ in i_option:
                    prompt = prompt_list[index_of_total]
                    
                    # Preprocess input for the model
                    single_input = self.processor(
                        text=prompt, images=_, padding="max_length", return_tensors="pt", max_length=77
                    ).to(self.device)
                    
                    # Create key mask for special token
                    keys = [torch.where(input_id == 32001, 1, 0) for input_id in single_input['input_ids']]

                    # Generate predictions based on specified method
                    if method == 'scaling_vis':
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
                            os.environ['SAVE_ATTN_PATH'] = f'{save_attn_dir_weight1}/{index_of_total}/'
                            os.makedirs(os.environ['SAVE_ATTN_PATH'], exist_ok=True)
                            output = self.model.generate(
                                **single_input, keys=keys, weight=weight1, 
                                max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                            )
                        else:
                            os.environ['SAVE_ATTN_PATH'] = f'{save_attn_dir_weight2}/{index_of_total}/'
                            os.makedirs(os.environ['SAVE_ATTN_PATH'], exist_ok=True)
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
                        threshold = 0.0004
                        print(f"\nUncertainty_prob: {uncertainty_prob}  |  Uncertainty_KL: {uncertainty_kl}  |  Threshold: {threshold}")
                        # Adjust attention based on uncertainty
                        if uncertainty_kl < threshold:
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
                    
                    elif method == 'adapt_vis_3':
                        change_greedy_to_add_weight()
                       
                        output = self.model.generate(
                            **single_input,weight=1.0,max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                        )
                        uncertainty_prob = self.get_uncertainty(output['scores'][0])
                        uncertainty_kl = self.get_uncertainty(output['scores'][0], method='kl_divergence')
                        threshold = 0.0004
                        print(f"\nUncertainty_prob: {uncertainty_prob}  |  Uncertainty_KL: {uncertainty_kl}  |  Threshold: {threshold}")
                        # Adjust attention based on uncertainty
                        if uncertainty_prob <= 0.4 and uncertainty_kl <= threshold:
                            output = self.model.generate(
                                **single_input, keys=keys, weight=weight1, 
                                max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                            )
                        elif (uncertainty_prob <= 0.4 and uncertainty_kl > threshold) or (uncertainty_prob > 0.4 and uncertainty_kl <= threshold):
                            pass # Use the original output 
                        else:
                            output = self.model.generate(
                                **single_input, keys=keys, weight=weight2, 
                                max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                            )
                        gen = self.processor.decode(output['sequences'][0][len(single_input['input_ids'][-1]):], skip_special_tokens=True)
                    
                    elif method == 'adapt_vis_4':
                        change_greedy_to_add_weight()
                       
                        output = self.model.generate(
                            **single_input,weight=1.0,max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                        )
                        uncertainty_prob = self.get_uncertainty(output['scores'][0])
                        uncertainty_kl = self.get_uncertainty(output['scores'][0], method='kl_divergence')
                        uncertainty = (uncertainty_prob + uncertainty_kl*1000) / 2
                        threshold = 0.4
                        print(f"\nUncertainty_prob: {uncertainty_prob}  |  Uncertainty_KL: {uncertainty_kl:06}  |  Uncertainty: {uncertainty}  |  Threshold: {threshold}")
                        # Adjust attention based on uncertainty
                        if uncertainty < threshold:
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
                    
                    elif method == 'adapt_vis_5':
                        change_greedy_to_add_weight()
                       
                        output = self.model.generate(
                            **single_input,weight=1.0,max_new_tokens=100, output_scores=True, return_dict_in_generate=True
                        )
                        uncertainty_prob = self.get_uncertainty(output['scores'][0], dataset=dataset)
                        uncertainty_kl = self.get_uncertainty(output['scores'][0], method='kl_divergence', dataset=dataset)
                        uncertainty = uncertainty_kl
                        print(f"\nUncertainty_prob: {uncertainty_prob}  |  Uncertainty_KL: {uncertainty_kl:06}  |  Uncertainty: {uncertainty}  |  Threshold: {threshold}")
                        # Adjust attention based on uncertainty
                        if uncertainty < threshold:
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
                        "uncertainty": uncertainty
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
            output_result_file_path = f'./output/results1.5_{dataset}_{method}_{weight}_{threshold}_{option}option_{TEST}.json'
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



