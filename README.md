# Detecção de Doenças em Laranja (Webcam USB)

Este projeto treina um modelo de classificação de doenças em laranja usando a pasta local `dataset/` (já separada em `train/`, `val/`, `test/`) e depois roda inferência em tempo real pela webcam.

## 1) Preparar ambiente

Pre-requisito importante:

- use `Python 3.10`, `3.11` ou `3.12` (TensorFlow nao suporta Python 3.14 no momento)

Exemplo com Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Se voce tiver apenas Python 3.14 instalado, instale o 3.11 e recrie o ambiente virtual.

## 2) Treinar o modelo

Treino base:

```bash
python train.py --data-dir dataset --epochs 20 --img-size 224 --batch-size 32
```

Treino com fine-tuning parcial (geralmente melhora):

```bash
python train.py --data-dir dataset --epochs 25 --img-size 224 --batch-size 32 --fine-tune
```

Arquivos gerados:

- `artifacts/orange_model.keras`
- `artifacts/labels.json`

## 3) Rodar com webcam USB

### Opcao A: inferencia simples

```bash
python webcam_infer.py --camera 0 --threshold 0.60
```

### Opcao B: webcam ao vivo + classificacao na tela (recomendada)

Este script mostra:

- imagem da webcam em tempo real
- retangulo central com a area analisada
- classificacao (`SAUDAVEL`, `DOENTE` ou `INCONCLUSIVO`)
- classe prevista + confianca
- preview do recorte usado no canto da tela

```bash
python webcam_live_classification.py --camera 0 --threshold 0.60
```

Se sua webcam USB nao estiver no indice `0`, tente `1` ou `2`:

```bash
python webcam_live_classification.py --camera 1
```

Atalhos:

- pressione `q` para sair

## Saída do sistema

Na tela, o modelo mostra:

- `SAUDAVEL` quando classe prevista é `healthy`
- `DOENTE` para as demais classes (`black_spot`, `canker`, `greening`, `rusttick`, `scab`)
- classe detalhada e confiança (`conf`)
- `INCONCLUSIVO` quando confianca < limiar (`--threshold`)

## Dicas para melhorar desempenho

- Use fundo mais limpo possível.
- Garanta boa iluminação, sem sombras fortes.
- Posicione a fruta ocupando boa parte do quadro.
- Colete imagens reais da sua câmera e faça re-treino periódico.
