pip install loralib diffusers==0.25.0 torch==2.0.1 xformers loralib fairscale
pip install open-clip-torch==2.20.0
pip install transformers==4.28.1
pip install --upgrade einops

python test_osediff.py \
-i preset/datasets/test_dataset/input \
-o preset/datasets/test_dataset/output \
--osediff_path OSEDIFF_PATH \
--pretrained_model_name_or_path SD21BASE_PATH \
--ram_ft_path DAPE_PATH \
--ram_path RAM_PATH


