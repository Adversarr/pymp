# Test if .venv exists, if not, create it and install requirements

WARNING_COLOR='\033[0;31m'
NO_COLOR='\033[0m'


if [ ! -d .venv ]; then
    echo -e "${WARNING_COLOR}No .venv directory found, creating one and installing requirements${NO_COLOR}"
    python3 --version
    if [ $? -ne 0 ]; then
        echo -e "${WARNING_COLOR}Python3 is not installed, please install it first${NO_COLOR}"
        exit 1
    fi

    python3 -m venv .venv --prompt=pymp-dev
    source .venv/bin/activate

    if [ $? -ne 0 ]; then
        echo -e "${WARNING_COLOR}Failed to create virtual environment${NO_COLOR}"
        exit 1
    fi
else
    source .venv/bin/activate
fi