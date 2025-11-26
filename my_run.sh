# Add the following line to enable test mode; otherwise, it defaults to validation mode
export TEST_MODE=True

python3 main_aro.py \
    --dataset Controlled_Images_A \
    --model-name='llava1.5' \
    --download \
    --method bidirectional_reasoning_2 \
    --weight1 0.5 \
    --weight2 1.5 \
    --threshold 0.4 \
    --option=four

#python3 main_aro.py --dataset=Controlled_Images_A --model-name='llava1.5' --download --method adapt_vis --weight1 0.5  --weight2 1.5 --threshold 0.4 --option=four