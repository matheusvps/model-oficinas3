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

## 2) Data augmentation (script separado)

Gera um novo dataset em disco (`dataset_aug/`) com:

- `train`: imagens originais + imagens aumentadas
- `val` e `test`: copiados sem alteracoes

```bash
python data_augmentation.py --input-dir dataset --output-dir dataset_aug --copies-per-image 2 --img-size 224
```

## 3) Treinar o modelo (script separado)

Treino base:

```bash
python train.py --data-dir dataset_aug --epochs 20 --img-size 224 --batch-size 32
```

Treino com fine-tuning parcial (geralmente melhora):

```bash
python train.py --data-dir dataset_aug --epochs 25 --img-size 224 --batch-size 32 --fine-tune
```

Arquivos gerados:

- `artifacts/orange_model.keras`
- `artifacts/labels.json`

### Como o treinamento foi feito

O script `train.py` usa **transfer learning** com `MobileNetV2` pretreinada em ImageNet para classificar 6 classes (`black_spot`, `canker`, `greening`, `healthy`, `rusttick`, `scab`).

Fluxo aplicado no treinamento:

1. Carrega os dados ja separados em `train/`, `val/` e `test/` com `image_dataset_from_directory`.
2. Redimensiona as imagens para `--img-size` (padrao `224x224`) e monta batches (`--batch-size`, padrao `32`).
3. Aplica o preprocessamento da MobileNetV2 (`mobilenet_v2.preprocess_input`).
4. Usa a base MobileNetV2 sem o topo (`include_top=False`) + `GlobalAveragePooling2D` + `Dropout(0.3)` + camada final `Dense` com `softmax`.
5. Na fase 1, treina com a base convolucional congelada (`base.trainable = False`).
6. Calcula `class_weight` automaticamente a partir da distribuicao de imagens por classe para reduzir desbalanceamento.
7. Treina com otimizador Adam (`--lr`, padrao `1e-4`), perda `sparse_categorical_crossentropy` e metrica `accuracy`.
8. Usa callbacks:
	- `EarlyStopping` monitorando `val_accuracy` (patience=5, restaura melhores pesos).
	- `ReduceLROnPlateau` monitorando `val_loss` (fator 0.3, patience=2).
9. Se `--fine-tune` estiver ativo, faz fase 2: descongela parte final da MobileNetV2 (ultimas ~40 camadas), reduz LR para `lr * 0.1` e treina por mais epocas.
10. Ao final, avalia em `val` e `test`, salva o modelo em `artifacts/orange_model.keras` e os metadados/metricas em `artifacts/labels.json`.

Observacao: o `labels.json` tambem gera um mapeamento binario (`healthy` vs `diseased`) para facilitar a classificacao final no uso com webcam.

## 4) Rodar com webcam USB

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

Para rodar: 
```bash
python webcam_live_classification.py --threshold 0.55 --inconclusive-margin 0.10 --min-margin 0.04 --stable-frames 5 --healthy-bias 0.08 --min-healthy-conf 0.30
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
