export TEST_MODE=False

python3 main_aro.py \
    --dataset Controlled_Images_A \
    --onunder 'true' \
    --model-name='llava1.5' \
    --download \
    --method adapt_vis_0.5 \
    --option=two