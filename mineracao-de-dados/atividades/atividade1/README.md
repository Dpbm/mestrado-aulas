# Atividade 1

## Como usar

Este projeto foi feito em ambiente `linux`, com as seguintes ferramentas:

- uv (https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)[https://docs.astral.sh/uv/getting-started/installation/#standalone-installer]
- make

Para ambientes `windows`, é recomendado o uso de [WSL](https://learn.microsoft.com/en-us/windows/wsl/about) ou ainda a execução via [Google Colab](https://colab.research.google.com/).

---

Para executar o código de **forma local**, primeiro adicione sua token do Kaggle no arquivo [key-setup-example.sh](./key-setup-example.sh), e execute os comandos:

```bash
uv sync
mv key-setup-example.sh key-setup.sh
make run
```

