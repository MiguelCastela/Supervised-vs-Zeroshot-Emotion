conda create -n emotion-nlp python=3.10 -y
conda activate emotion-nlp

conda install pytorch torchvision torchaudio -c pytorch -y
conda install -c conda-forge transformers datasets scikit-learn -y
python -m pip install sentence-transformers
