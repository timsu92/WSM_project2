FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

# python
ENV PYTHONUNBUFFERED=1 \
  # prevents python creating .pyc files
  PYTHONDONTWRITEBYTECODE=1 \
  \
  # pip
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  \
  # poetry
  # https://python-poetry.org/docs/configuration/#using-environment-variables
  POETRY_VERSION=1.8.4 \
  # make poetry install to this location
  POETRY_HOME="/opt/poetry" \
  # make poetry create the virtual environment in the project's root
  # it gets named `.venv`
  POETRY_VIRTUALENVS_IN_PROJECT=true \
  # do not ask any interactive question
  POETRY_NO_INTERACTION=1 \
  \
  # paths
  # this is where our requirements + virtual environment will live
  PYSETUP_PATH="/workspace/project2" \
  VENV_PATH="/workspace/project2/.venv"

# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# APT
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
	&& apt-get install -y \
  # deps for installing poetry
  curl \
  # deps for building python deps
  python3 build-essential \
  # project deps
  default-jre default-jdk \
	&& rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://install.python-poetry.org | python3 -

# copy project requirement files here to ensure they don't change
WORKDIR $PYSETUP_PATH

# setup shell completion for poetry
RUN case "$0" in \
  *bash*) echo "setup Poetry shell completion for bash" && poetry completions bash > ${XDG_DATA_HOME:-~/.local/share}/bash-completion/completions/poetry ;; \
  *zsh*) echo "setup Poetry shell completion for zsh" && poetry completions zsh > "${fpath[1]}/_poetry" ;; \
  *fish*) echo "setup Poetry shell completion for fish" && poetry completions fish > ~/.config/fish/completions/poetry.fish ;; \
  *) echo "Poetry completions are not supported in your shell" ;; \
  esac
