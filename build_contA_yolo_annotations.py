import os
import json
import argparse

from PIL import Image
from ultralytics import YOLO
import re
from typing import List, Dict, Any, Optional, Tuple
import string

QUESTION_PATTERNS = [
    # Where is the beer bottle in relation to the armchair?
    re.compile(
        r"the (.+?) in relation to the (.+?)\?\s*answer with",
        re.IGNORECASE,
    ),
    # 만약 'Where is the A in relation to the B in the image?' 같은 변형이 있다면 패턴 추가
]

PHRASE_TO_YOLO_CANDIDATES = {
    "beer bottle": ["bottle", "wine bottle"],
    "bottle": ["bottle", "wine bottle"],
    "armchair": ["armchair", "chair", "couch", "sofa"],
    "chair": ["chair", "armchair"],
    "couch": ["couch", "sofa"],
    "sofa": ["couch", "sofa"],
    "person": ["person", "man", "woman"],
    "man": ["person"],
    "woman": ["person"],
    "cat": ["cat"],
    "dog": ["dog"],
    "fridge": ["refrigerator"],
    "refrigerator": ["refrigerator"],
    # ...
}

def get_yolo_class_candidates_from_phrase(phrase: str) -> list[str]:
    norm = normalize_phrase(phrase)
    # 정확히 매핑이 있으면 그대로
    if norm in PHRASE_TO_YOLO_CANDIDATES:
        return PHRASE_TO_YOLO_CANDIDATES[norm]
    # 없는 경우, 단어 단위 fallback (예: "yellow armchair" -> "armchair")
    tokens = norm.split()
    for t in tokens[::-1]:  # 뒤에서부터(핵심 명사가 보통 뒤에 옴)
        if t in PHRASE_TO_YOLO_CANDIDATES:
            return PHRASE_TO_YOLO_CANDIDATES[t]
    # 최종: YOLO class name과의 부분 문자열 매칭에 맡기도록 빈 리스트 반환
    return []

def normalize_phrase(s: str) -> str:
    s = s.lower()
    # 구두점 제거
    s = s.translate(str.maketrans("", "", string.punctuation))
    # 다중 공백 정리
    s = " ".join(s.split())
    return s

def extract_objects_from_question(question: str) -> Tuple[str, str]:
    """
    question에서 (obj_a, obj_b)를 추출.
    예: "Where is the beer bottle in relation to the armchair? Answer with ..."
      -> ("beer bottle", "armchair")
    """
    q = question.strip()
    for pat in QUESTION_PATTERNS:
        m = pat.search(q)
        if m:
            obj_a = m.group(1).strip()
            obj_b = m.group(2).strip()
            return obj_a, obj_b
    raise ValueError(f"Cannot parse objects from question: {question}")

def load_cont_a_raw(path: str) -> List[Dict[str, Any]]:
    """
    입력: 각 줄이 다음 형식인 jsonl:
      {
        "image_path": "images/contA/0001.png",
        "question": "Where is the beer bottle in relation to the armchair?",
        "answer_label": "left"
      }
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def run_yolo_on_image(
    model: YOLO,
    image_path: str,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_dets: int = 100,
) -> List[Dict[str, Any]]:
    """
    한 장의 이미지에 YOLO detection을 돌리고, 박스 정보 리스트를 반환.
    각 박스는 {
      "x1", "y1", "x2", "y2", "cls_id", "cls_name", "conf"
    } 형태.
    """
    results = model(
        image_path,
        conf=conf_thres,
        iou=iou_thres,
        max_det=max_dets,
        verbose=False,
    )

    boxes_out: List[Dict[str, Any]] = []
    for r in results:
        # r: ultralytics.engine.results.Results
        if r.boxes is None or len(r.boxes) == 0:
            continue

        for b in r.boxes:
            # xyxy: [x1, y1, x2, y2]
            xyxy = b.xyxy[0].tolist()
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            conf = float(b.conf[0].item())
            cls_id = int(b.cls[0].item())
            cls_name = str(r.names[cls_id])

            boxes_out.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "conf": conf,
                }
            )

    return boxes_out


def filter_boxes_by_question(
    boxes: List[Dict[str, Any]],
    question: str,
    min_conf: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    (옵션) 질문에 등장하는 단어와 YOLO 클래스명을 간단히 매칭해,
    관련성이 높아 보이는 박스만 남기는 필터.

    - 예: question: "Where is the beer bottle in relation to the armchair?"
      YOLO cls_name: "bottle", "chair"
      -> 'bottle', 'chair' 토큰이 question에 들어 있으면 keep.

    너무 aggressive하면 놓칠 수 있으니, 기본은 '전체 박스 사용'으로 두고,
    후처리/분석에서 다시 골라도 됨.
    """
    q = question.lower()
    kept = []
    for box in boxes:
        if box["conf"] < min_conf:
            continue
        name = box["cls_name"].lower()
        # 공백 기준으로 나눈 토큰 중 하나라도 question에 등장하면 keep
        tokens = name.replace("-", " ").split()
        if any(tok in q for tok in tokens):
            kept.append(box)
    # fallback: 아무것도 안 남았으면 원본 boxes 반환
    return kept if kept else boxes

def score_phrase_class_match(phrase: str, cls_name: str) -> float:
    """
    fallback용 간단 스코어: 토큰 겹치는 수 기반.
    필요하면 더 정교한 스코어링(예: embedding)으로 교체 가능.
    """
    p_tokens = set(normalize_phrase(phrase).split())
    c_tokens = set(normalize_phrase(cls_name).split())
    if not p_tokens or not c_tokens:
        return 0.0
    return len(p_tokens & c_tokens) / len(p_tokens | c_tokens)


def select_boxes_for_phrase(
    boxes: List[Dict[str, Any]],   # run_yolo_on_image의 출력
    phrase: str,
) -> List[Dict[str, Any]]:
    """
    한 객체 phrase에 해당하는 YOLO 박스 선택.
    1) 사전 기반 후보 클래스 필터
    2) 없으면 토큰 overlap 스코어가 가장 높은 클래스 하나를 골라 그 클래스의 박스들만 사용
    """
    cand_classes = get_yolo_class_candidates_from_phrase(phrase)

    # 1) 후보 클래스가 있으면 그 중에 cls_name이 있는 박스만
    if cand_classes:
        cand_set = {c.lower() for c in cand_classes}
        matched = [b for b in boxes if b["cls_name"].lower() in cand_set]
        if matched:
            # confidence 내림차순으로 정렬해서 우선순위 줄 수 있음
            matched.sort(key=lambda b: b["conf"], reverse=True)
            return matched

    # 2) 후보 클래스가 없거나 매칭 실패 -> fallback
    #    phrase와 클래스 이름 간 토큰 overlap이 가장 큰 cls_name을 선택
    if not boxes:
        return []

    # cls_name마다 phrase와의 스코어 계산
    best_cls = None
    best_score = 0.0
    for b in boxes:
        cls_name = b["cls_name"]
        s = score_phrase_class_match(phrase, cls_name)
        if s > best_score:
            best_score = s
            best_cls = cls_name

    if best_cls is None or best_score == 0.0:
        # 완전히 매칭 안 되면, 차라리 비우고 downstream에서 "missing"으로 처리
        return []

    matched = [b for b in boxes if b["cls_name"].lower() == best_cls.lower()]
    matched.sort(key=lambda b: b["conf"], reverse=True)
    return matched

def build_annotations_with_objects(
    raw_ann_path: str,
    image_root: str,
    yolo_model_path: str,
    out_path: str,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_dets: int = 100,
):
    data = load_cont_a_raw(raw_ann_path)
    model = YOLO(yolo_model_path)

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(data):
            rel_img_path = item["image_path"]
            img_path = os.path.join(image_root, rel_img_path)
            question = item["question"]

            try:
                obj_a_phrase, obj_b_phrase = extract_objects_from_question(question)
            except ValueError as e:
                print(f"[WARN] cannot parse question #{idx}: {e}")
                obj_a_phrase, obj_b_phrase = None, None

            try:
                _ = Image.open(img_path)
            except Exception as e:
                print(f"[WARN] cannot open image: {img_path} ({e})")
                new_item = dict(item)
                new_item["obj_a"] = obj_a_phrase
                new_item["obj_b"] = obj_b_phrase
                new_item["yolo_a_boxes"] = []
                new_item["yolo_b_boxes"] = []
                fout.write(json.dumps(new_item) + "\n")
                continue

            # YOLO 전체 박스
            all_boxes = run_yolo_on_image(
                model,
                img_path,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_dets=max_dets,
            )

            if obj_a_phrase is not None:
                yolo_a = select_boxes_for_phrase(all_boxes, obj_a_phrase)
            else:
                yolo_a = []

            if obj_b_phrase is not None:
                yolo_b = select_boxes_for_phrase(all_boxes, obj_b_phrase)
            else:
                yolo_b = []

            new_item = dict(item)
            new_item["obj_a"] = obj_a_phrase
            new_item["obj_b"] = obj_b_phrase
            new_item["yolo_a_boxes"] = yolo_a
            new_item["yolo_b_boxes"] = yolo_b

            fout.write(json.dumps(new_item) + "\n")

            if (idx + 1) % 50 == 0:
                print(f"[{idx+1}/{len(data)}] processed")



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_ann", type=str, required=True,
                    help="Cont_A raw annotation jsonl (no yolo_boxes)")
    ap.add_argument("--image_root", type=str, required=True,
                    help="Root directory for Cont_A images")
    ap.add_argument("--yolo_model", type=str, required=True,
                    help="YOLO model path or name, e.g. 'yolov8x.pt'")
    ap.add_argument("--out", type=str, required=True,
                    help="Output jsonl with yolo_boxes")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max_dets", type=int, default=100)
    ap.add_argument("--filter_by_question", action="store_true",
                    help="Filter YOLO boxes using question text (optional)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_annotations_with_objects(
        raw_ann_path=args.raw_ann,
        image_root=args.image_root,
        yolo_model_path=args.yolo_model,
        out_path=args.out,
        conf_thres=args.conf,
        iou_thres=args.iou,
        max_dets=args.max_dets,
    )
