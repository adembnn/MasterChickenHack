B=/p/scratch/training2666/destors1/MasterChickenHack/unlearning; D=$B/adem/t3-nonmembres; cd $D/code
touch $D/logs/hold
HF_HOME=$D/hf $D/.venv/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('stabilityai/stable-diffusion-xl-base-1.0', allow_patterns=['vae_1_0/config.json','vae_1_0/diffusion_pytorch_model.fp16.safetensors','vae/*.json']))" 2>&1 | tail -1
HF_HOME=$D/hf HF_HUB_OFFLINE=1 $D/.venv/bin/python -c "
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', vae=vae, torch_dtype=torch.float16, variant='fp16'); print('SDXL_LOAD_OK')
pipe.load_ip_adapter('h94/IP-Adapter', subfolder='sdxl_models', weight_name='ip-adapter-plus_sdxl_vit-h.safetensors', image_encoder_folder='models/image_encoder'); print('IPA_LOAD_OK')" 2>&1 | grep -E "OK|Error|error" | tail -4
nohup srun --jobid=15902436 --overlap --ntasks=1 --cpus-per-task=24 --gres=gpu:4 bash $D/code/fix1.sh > $D/logs/fix1.log 2>&1 < /dev/null &
sleep 25; cat $D/logs/fix1.log | tail -8; tail -n 2 $D/logs/sdxl_i2i_50.log | cut -c1-300
