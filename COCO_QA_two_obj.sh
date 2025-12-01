export TEST_MODE=True

python3 main_aro.py \
    --dataset COCO_QA_two_obj \
    --model-name='llava1.5' \
    --download \
    --method adapt_vis_entropy \
    --weight1 0.5 \
    --weight2 2.0 \
    --threshold 0.95 \
    --option=four