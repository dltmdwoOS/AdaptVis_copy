TEST_MODE=False
DATASET="Controlled_Images_A"      # 필요에 따라 변경
LEN_OPTION="four"             # 필요에 따라 변경
MODEL_NAME="llava1.5"
METHOD="adapt_vis_research"

OUT_DIR="output"
RESULT_JSON="${OUT_DIR}/results1.5_${DATASET}_${METHOD}_1.0_1.0_${LEN_OPTION}option_${TEST_MODE}.json"
SAVE_PREFIX="${OUT_DIR}/analysis_${MODEL_NAME}_${DATASET}_${METHOD}"

# 결과 분석 + 그래프 생성 + 최적 (weight1, weight2) 쌍과 최적 threshold 도출

python3 research.py \
  --results-path "${RESULT_JSON}" \
  --baseline-weight "1.0" \
  --save-prefix "${SAVE_PREFIX}"