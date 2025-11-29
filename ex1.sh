export TEST_MODE=True

python3 main_aro.py \
    --dataset Controlled_Images_A \
    --model-name='llava1.5' \
    --download \
    --method adapt_vis \
    --weight1 0.5 \
    --weight2 1.2 \
    --threshold 0.4 \
    --option=four

# python3 main_aro.py \
#     --dataset Controlled_Images_A \
#     --model-name='llava1.5' \
#     --download \
#     --method adapt_vis_5 \
#     --weight1 0.5 \
#     --weight2 1.2 \
#     --threshold 0.05 \
#     --option=four


# python3 main_aro.py \
#     --dataset VG_QA_two_obj \
#     --model-name='llava1.5' \
#     --download \
#     --method adapt_vis_5 \
#     --weight1 0.5 \
#     --weight2 1.2 \
#     --threshold 0.05 \
#     --option=six