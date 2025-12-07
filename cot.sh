export TEST_MODE=False

python3 main_aro.py \
    --dataset Controlled_Images_A \
    --model-name='llava1.5' \
    --download \
    --method few_shot_CoT \
    --option=four