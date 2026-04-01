#!/bin/bash

# vars
repo="https://github.com/charly-chrtx/maestro.git"
dir="/opt/maestro"

# apt
sudo apt update
sudo apt install -y git python3-pip python3-venv python3-dev build-essential cmake portaudio19-dev curl

# git
if [ -d "$dir/.git" ]; then
    sudo git -C "$dir" pull
else
    sudo git clone "$repo" "$dir"
fi

# perms
sudo chown -R $USER:$USER "$dir"
cd "$dir" || exit

# venv
python3 -m venv venv
source venv/bin/activate

# pip
pip install --upgrade pip wheel
pip install -r requirements.txt

# ollama
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

# bin
echo '#!/bin/bash' | sudo tee /usr/local/bin/maestro > /dev/null
echo 'cd /opt/maestro && /opt/maestro/venv/bin/python maestro.py "$@"' | sudo tee -a /usr/local/bin/maestro > /dev/null
sudo chmod +x /usr/local/bin/maestro