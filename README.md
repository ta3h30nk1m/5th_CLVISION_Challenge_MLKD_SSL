# installation
- Additional library required: kornia
```bash
git clone https://github.com/ta3h30nk1m/clvision24-custom.git
cd clvision-challenge-2024
conda create -n clvision24 python==3.10
conda activate clvision24
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install avalanche-lib==0.4.0
pip install kornia 
```

# How to run
- run in background(nohup)
```bash
./run.sh $gpu_id $scenario_num $run_name
```