brew install pyenv

export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

source ~/.bashrc   # 或者 source ~/.zshrc

pyenv install 3.9.18


brew install pip

pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple


pip3 install -r requirements.txt

