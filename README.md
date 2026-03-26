# Detecção de Doenças em Laranja (Webcam USB)

Este projeto treina um modelo de classificação de doenças em laranja usando a pasta local `dataset/` (já separada em `train/`, `val/`, `test/`) e depois roda inferência em tempo real pela webcam.

## Fontes dos datasets

Os dados utilizados neste projeto vieram de trabalhos publicados no Mendeley Data:

1. Orange Fruit Diseases Dataset (Mendeley):
	- https://data.mendeley.com/datasets/6szsnpypdd/1/files/2fcace93-0985-4b8a-aec4-45ba4ca1752c
2. An Image Dataset of Citrus Fruit and Leaves for Detection and Classification of Diseases:
	- https://data.mendeley.com/datasets/3f83gxmv57/1
3. In-Field Citrus Disease Classification via Convolutional Neural Network from Smartphone Images:
	- https://data.mendeley.com/datasets/3f83gxmv57/1

Observacao: os itens 2 e 3 apontam para o mesmo link fornecido acima.

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

### Visao geral do pipeline

- `data_augmentation.py`: cria `dataset_aug/`, copia `val/test` sem alterar e gera variacoes de `train` em disco.
- `train.py`: faz treino multiclasse (6 classes), validacao, teste e exporta modelo + metadados.
- `webcam_infer.py` e `webcam_live_classification.py`: carregam os artefatos para inferencia em tempo real.

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

### Detalhamento tecnico do fluxo de treino

#### 1) Entrada de dados e formatacao

- O `image_dataset_from_directory` infere os rótulos pelo nome das pastas.
- `label_mode="int"` produz inteiros para usar com `sparse_categorical_crossentropy`.
- O shape de entrada e padronizado para `(img_size, img_size, 3)`.
- O treino usa `shuffle=True` com `seed=42` para reproducibilidade basica.
- `val` e `test` usam `shuffle=False` para manter avaliacao deterministica.

#### 2) Pipeline eficiente com `tf.data`

- Depois de criar os datasets, o script aplica `prefetch(tf.data.AUTOTUNE)`.
- Isso permite preparar o proximo batch em paralelo ao processamento do batch atual na GPU/CPU.
- Em pratica, essa etapa reduz tempo ocioso durante o treino.

#### 3) Normalizacao correta para MobileNetV2

- O preprocessamento `mobilenet_v2.preprocess_input` converte os pixels da faixa `[0, 255]` para a faixa esperada pela rede base.
- Essa compatibilidade e importante para aproveitar bem os pesos pretreinados em ImageNet.

#### 4) Arquitetura de classificacao

- Backbone: `MobileNetV2(include_top=False, weights="imagenet")`.
- Cabeca customizada:
	- `GlobalAveragePooling2D` para reduzir mapas de ativacao em vetor.
	- `Dropout(0.3)` para reduzir overfitting.
	- `Dense(num_classes, activation="softmax")` para classificacao multiclasse.

Essa combinacao e um padrao eficiente quando o dataset e menor que ImageNet e o objetivo e adaptar rapidamente o modelo para um dominio especifico.

#### 5) Balanceamento com `class_weight`

- O script conta imagens por classe em `train/`.
- Calcula peso por classe com a formula:
	- `peso = total_imagens / (num_classes * imagens_da_classe)`
- Classes com menos exemplos recebem maior peso na perda.

Isso ajuda a evitar que o modelo favoreca apenas classes mais frequentes.

#### 6) Treino em duas fases

- Fase 1 (sempre): `base.trainable = False`.
	- A MobileNetV2 funciona como extrator de caracteristicas fixo.
	- Treina apenas a cabeca final.
- Fase 2 (opcional com `--fine-tune`):
	- Descongela a base e mantem congeladas todas as camadas exceto as ~40 finais.
	- Recompila com `learning_rate` menor (`lr * 0.1`).
	- Executa mais epocas (`max(4, epochs // 2)`).

Esse desenho reduz risco de destruir o conhecimento pretreinado e tende a melhorar performance final.

#### 7) Callbacks e criterio de parada

- `EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)`:
	- interrompe quando o modelo para de melhorar em validacao.
	- volta para os melhores pesos vistos durante o treino.
- `ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6)`:
	- reduz automaticamente a taxa de aprendizado quando a perda de validacao estagna.

Juntos, esses callbacks ajudam a estabilizar convergencia e reduzir overfitting.

#### 8) Avaliacao final e artefatos

- O script avalia e imprime metricas de `val` e `test` (`loss` e `accuracy`).
- Salva `artifacts/orange_model.keras` com a arquitetura + pesos.
- Salva `artifacts/labels.json` com:
	- `img_size`
	- `class_names`
	- `binary_map` (indice -> `healthy`/`diseased`)
	- `train_counts`
	- `val_metrics` e `test_metrics`

Esse `labels.json` permite que os scripts de webcam usem exatamente o mesmo mapeamento de classes do treino.

## Bibliotecas usadas e o que abstraem

- `tensorflow` / `tf.keras`:
	- camadas (`layers`), modelo (`Model`), otimizador (`Adam`), callbacks e carregamento de dataset por diretorio.
	- abstrai grande parte do ciclo de treino (`fit`, `evaluate`, `save`) sem precisar implementar loop manual.
- `numpy`:
	- manipulacao de arrays e controle de seed auxiliar.
- `opencv-python (cv2)`:
	- captura de webcam, desenho de overlays e processamento de imagem em tempo real.
- `argparse` (stdlib):
	- cria interface de linha de comando com parametros (`--epochs`, `--threshold`, etc.).
- `pathlib` (stdlib):
	- manipula caminhos de forma mais segura e legivel.
- `json` (stdlib):
	- serializa metadados e resultados em `labels.json`.
- `shutil` (stdlib, no augmentation):
	- copia/recria estrutura de pastas para gerar o dataset aumentado.

Em resumo, o projeto usa `tf.keras` para abstrair o aprendizado profundo e `OpenCV` para a camada de visao em tempo real, mantendo scripts curtos e focados na logica principal.

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
