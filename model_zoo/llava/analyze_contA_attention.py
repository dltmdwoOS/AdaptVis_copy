# analyze_contA_attention.py
import os
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2

# 1) 모델/프로세서 임포트
#   - HF로 변환된 모델을 쓰면 transformers에서 바로 import
#   - 논문 레포의 로컬 구현을 쓰면 아래와 같이 상대 import (경로에 맞게 수정)
#
# from transformers import LlavaForConditionalGeneration, LlavaProcessor
from modeling_llava import LlavaForConditionalGeneration  # repo 경로에 맞게 수정
from processing_llava import LlavaProcessor                # repo 경로에 맞게 수정


IMAGE_TOKEN_ID = 32000  # LLaVA에서 <image> special token id (레포 설정에 맞게 확인 필요)
NUM_IMAGE_TOKENS = 24 * 24  # 논문에서 사용하는 CLIP 24x24 패치 수 [1]


def load_cont_a_annotations(path: str) -> List[Dict[str, Any]]:
    """
    Cont_A용 어노테이션 로더 예시.
    형식 예시 (jsonl):
      {
        "image_path": "images/contA/0001.png",
        "question": "Where is the beer bottle in relation to the armchair?",
        "answer_label": "left",              # GT 라벨
        "yolo_boxes": [                      # YOLO bbox 리스트 (x1, y1, x2, y2, class_name)
          {"x1": 50, "y1": 40, "x2": 120, "y2": 200, "cls": "beer bottle"},
          {"x1": 150, "y1": 60, "x2": 260, "y2": 240, "cls": "armchair"}
        ]
      }
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
    return data


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------
#  Attention 관련 유틸
# -------------------------------

def _get_image_token_indices(input_ids: torch.Tensor) -> List[torch.Tensor]:
    """
    각 시퀀스별로 최종 LM 시퀀스에서 이미지 토큰 위치 인덱스(정수 텐서)를 리턴.
    전제: input_ids 안의 IMAGE_TOKEN_ID 한 개가 LM 내부에서 576개 이미지 토큰으로 확장됨.
    - input_ids: [B, T_text]
    - LM 시퀀스 길이: T_text - 1 + NUM_IMAGE_TOKENS
    """
    bsz, t_text = input_ids.shape
    image_token_indices = []

    # <image> 토큰의 위치 (text space)
    for b in range(bsz):
        ids = input_ids[b]
        image_pos = (ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        assert len(image_pos) == 1, "현재 코드는 이미지 1장(<image> 토큰 1개)만 가정"
        image_pos = image_pos.item()

        # LM 시퀀스에서의 인덱스:
        #  - [0 .. image_pos-1]: 텍스트
        #  - [image_pos .. image_pos+NUM_IMAGE_TOKENS-1]: 이미지 패치
        #  - 이후: 나머지 텍스트
        start = image_pos
        end = image_pos + NUM_IMAGE_TOKENS
        img_idx = torch.arange(start, end, device=input_ids.device)
        image_token_indices.append(img_idx)

    return image_token_indices


def _extract_attn_to_image_tokens(
    attentions: Tuple[torch.Tensor, ...],
    image_token_indices: List[torch.Tensor],
    answer_positions: torch.Tensor,
    target_layers: List[int],
) -> Dict[int, np.ndarray]:
    """
    각 타겟 레이어에서 '마지막 토큰(=answer_positions)'이 이미지 토큰에 주는 attention을 추출.
    - attentions[l]: [B, H, T, T]
    - image_token_indices[b]: 길이 NUM_IMAGE_TOKENS
    - answer_positions: [B] (LM 시퀀스 상의 위치)
    리턴: {layer_idx: np.array [B, NUM_IMAGE_TOKENS]} (head 평균)
    """
    layer_attn_maps = {}
    for l in target_layers:
        attn_l = attentions[l]  # [B, H, T, T]
        bsz, n_heads, T, _ = attn_l.shape
        maps = []
        for b in range(bsz):
            pos = answer_positions[b].item()
            # [H, T] -> 이미지 토큰 부분만 슬라이스 -> [H, num_img]
            attn_vec = attn_l[b, :, pos, :]  # [H, T]
            img_idx = image_token_indices[b]  # [num_img]
            attn_img = attn_vec[:, img_idx]  # [H, num_img]
            # head 평균 -> [num_img]
            attn_img_mean = attn_img.mean(dim=0)  # [num_img]
            maps.append(attn_img_mean.cpu().numpy())
        layer_attn_maps[l] = np.stack(maps, axis=0)  # [B, num_img]
    return layer_attn_maps


def _attn_vec_to_patch_map(attn_vec: np.ndarray, grid_size: int = 24) -> np.ndarray:
    """
    길이 576 벡터를 [24, 24] 패치 맵으로 reshape.
    """
    assert attn_vec.shape[-1] == grid_size * grid_size
    patch_map = attn_vec.reshape(grid_size, grid_size)
    # 음수 있을 수 있으니 ReLU 후 정규화
    patch_map = np.maximum(patch_map, 0.0)
    if patch_map.sum() > 0:
        patch_map = patch_map / (patch_map.sum() + 1e-6)
    return patch_map


def overlay_attention_on_image(
    image: Image.Image,
    patch_map: np.ndarray,
    alpha: float = 0.5,
    cmap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """
    24x24 patch_map을 원본 해상도로 업샘플링 후 heatmap overlay.
    """
    img_w, img_h = image.size
    heat = patch_map
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    heat = cv2.resize(heat.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    heat_uint8 = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cmap)

    img_np = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    overlay_bgr = cv2.addWeighted(img_bgr, 1 - alpha, heat_color, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_rgb)


# -------------------------------
#  YOLO overlap 관련
# -------------------------------

def _boxes_to_patch_mask(
    bboxes: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    grid_size: int = 24,
) -> np.ndarray:
    """
    YOLO bbox (x1, y1, x2, y2)를 patch grid (24x24) 바이너리 마스크로 변환.
    """
    mask = np.zeros((grid_size, grid_size), dtype=np.float32)

    for box in bboxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        # bbox를 패치 인덱스 범위로 변환
        px1 = int(np.floor(x1 / img_w * grid_size))
        py1 = int(np.floor(y1 / img_h * grid_size))
        px2 = int(np.ceil(x2 / img_w * grid_size))
        py2 = int(np.ceil(y2 / img_h * grid_size))

        px1 = np.clip(px1, 0, grid_size - 1)
        py1 = np.clip(py1, 0, grid_size - 1)
        px2 = np.clip(px2, 1, grid_size)
        py2 = np.clip(py2, 1, grid_size)

        mask[py1:py2, px1:px2] = 1.0

    if mask.sum() > 0:
        mask = mask / (mask.sum() + 1e-6)
    return mask


def cosine_overlap(att_map: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """
    논문과 같은 cosine similarity 기반 overlap 지표.[attached_file:1]
    """
    if mask.sum() == 0:
        return None
    a = att_map.reshape(-1)
    m = mask.reshape(-1)
    # 이미 각자 sum=1 정규화된 상태라고 가정
    num = float((a * m).sum())
    den = float(np.linalg.norm(a) * np.linalg.norm(m) + 1e-6)
    return num / den


# -------------------------------
#  메인 루프
# -------------------------------

def run_analysis(
    model_name_or_path: str,
    cont_a_ann_path: str,
    image_root: str,
    yolo_use_all_boxes: bool,
    out_json_path: str,
    out_vis_dir: str,
    target_layers: List[int],
    max_samples: Optional[int] = None,
):
    device = get_device()
    os.makedirs(out_vis_dir, exist_ok=True)

    # HF Hub에 올려둔 모델이면 transformers에서, 아니면 로컬 파일에서 로딩
    processor = LlavaProcessor.from_pretrained(model_name_or_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    data = load_cont_a_annotations(cont_a_ann_path)
    if max_samples is not None:
        data = data[:max_samples]

    results = []

    for idx, item in enumerate(data):
        image_path = os.path.join(image_root, item["image_path"])
        question = item["question"]
        gt_label = item.get("answer_label")
        yolo_boxes = item.get("yolo_boxes", [])

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        # 1) 입력 텐서 생성
        prompt = question  # 이미 "Answer with left, right, on or under." 형식이라고 가정 [attached_file:1]
        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        ).to(device)

        input_ids = inputs["input_ids"]  # [1, T_text]
        image_token_indices = _get_image_token_indices(input_ids)

        # 2) 답 생성 (greedy)
        with torch.no_grad():
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # 디코딩 + 간단한 답 파싱
        generated_ids = gen_outputs.sequences  # [1, T_total]
        answer_text = processor.tokenizer.decode(
            generated_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip().lower()

        # 라벨 space에 맞춰 정규화 (left/right/on/under 중 하나로 매핑)
        def normalize_answer(ans: str) -> str:
            for k in ["left", "right", "on", "under", "above", "below"]:
                if k in ans:
                    if k == "above":
                        return "on"
                    if k == "below":
                        return "under"
                    return k
            return ans

        pred_label = normalize_answer(answer_text)

        # 정답 여부
        is_correct = None
        if gt_label is not None:
            is_correct = (pred_label == gt_label)

        # 마지막 생성 토큰의 confidence (softmax 확률)[attached_file:1]
        last_step_scores = gen_outputs.scores[-1]  # [1, V]
        last_probs = last_step_scores.softmax(dim=-1)
        last_token_id = generated_ids[0, -1]
        confidence = float(last_probs[0, last_token_id].item())

        # 3) attention 추출을 위한 full forward
        #    - 입력: 질문 + 모델이 생성한 답까지 모두 포함한 시퀀스
        with torch.no_grad():
            full_inputs = {
                "input_ids": generated_ids.to(device),
                "attention_mask": torch.ones_like(generated_ids, device=device),
                "pixel_values": inputs["pixel_values"],
            }
            outputs = model(
                **full_inputs,
                output_attentions=True,
                return_dict=True,
            )

        attentions = outputs.attentions  # Tuple[L] each [1, H, T, T] [attached_file:2]
        # 마지막 토큰 위치 (시퀀스 끝 - 1)
        answer_positions = torch.tensor([generated_ids.shape[1] - 1], device=device)

        layer_attn_vecs = _extract_attn_to_image_tokens(
            attentions=attentions,
            image_token_indices=image_token_indices,
            answer_positions=answer_positions,
            target_layers=target_layers,
        )

        sample_record = {
            "idx": idx,
            "image_path": item["image_path"],
            "question": question,
            "gt_label": gt_label,
            "pred_label": pred_label,
            "is_correct": is_correct,
            "confidence": confidence,
            "layers": {},
        }

        for l in target_layers:
            attn_vec = layer_attn_vecs[l][0]  # [num_img] for this sample
            patch_map = _attn_vec_to_patch_map(attn_vec, grid_size=24)

            # YOLO bbox 사용 범위 선택 (예: 두 객체 전부 사용)
            if yolo_use_all_boxes:
                bboxes = yolo_boxes
            else:
                # 필요하다면 GT relation에서 어떤 객체만 고르는 로직 추가
                bboxes = yolo_boxes

            mask = _boxes_to_patch_mask(bboxes, img_w=img_w, img_h=img_h, grid_size=24)
            overlap = cosine_overlap(patch_map, mask)

            # 시각화 저장
            vis_img = overlay_attention_on_image(image, patch_map, alpha=0.5)
            vis_filename = f"contA_{idx:05d}_layer{l}.png"
            vis_path = os.path.join(out_vis_dir, vis_filename)
            vis_img.save(vis_path)

            sample_record["layers"][str(l)] = {
                "vis_path": vis_filename,
                "overlap_cosine": overlap,
                "image_attention_sum": float(attn_vec.sum()),  # 필요시
            }

        results.append(sample_record)

        if (idx + 1) % 50 == 0:
            print(f"[{idx+1}/{len(data)}] processed")

    # 4) JSONL로 저장
    with open(out_json_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help="HF Llava model path or name")
    ap.add_argument("--cont_a_ann", type=str, required=True,
                    help="Cont_A annotation jsonl path")
    ap.add_argument("--image_root", type=str, required=True,
                    help="Root dir for Cont_A images")
    ap.add_argument("--out_json", type=str, required=True,
                    help="Output JSONL path")
    ap.add_argument("--out_vis_dir", type=str, required=True,
                    help="Dir to save attention overlay images")
    ap.add_argument("--layers", type=int, nargs="+", default=[17],
                    help="Layer indices to visualize (0-based)")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--yolo_use_all_boxes", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        model_name_or_path=args.model,
        cont_a_ann_path=args.cont_a_ann,
        image_root=args.image_root,
        yolo_use_all_boxes=args.yolo_use_all_boxes,
        out_json_path=args.out_json,
        out_vis_dir=args.out_vis_dir,
        target_layers=args.layers,
        max_samples=args.max_samples,
    )
