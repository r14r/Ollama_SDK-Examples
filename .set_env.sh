export ENV_NAME=Default
export ENV_HOME=$PWD
export VSCODE_ENV=src.ollama-examples

. venv env        init
. venv vscode     init

. venv python     init 3.11
