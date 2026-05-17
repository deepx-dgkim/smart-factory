# smart-factory

## Synthetic DEEPX can defect segmentation dataset

가상의 `DEEPX` 통조림 캔 표면 결함을 Ultralytics YOLO26 segmentation으로 학습할 수 있도록 합성 데이터를 생성합니다. 캔 제품 크기와 위치는 모든 이미지에서 고정이고, 이미지는 항상 `1280x1280`입니다.

정상 제품 이미지는 `assets/deepx_can_reference_01.png`부터 `assets/deepx_can_reference_04.png`까지 4개 레퍼런스를 1280x1280으로 정규화해서 사용합니다. 각 샘플은 현재 조도를 100으로 볼 때 `90~110` 범위 안에서만 조명이 달라지고, defect는 정상 이미지 위에 합성됩니다.

### Defect classes

| id | class | description |
| --- | --- | --- |
| 0 | `no_defect` | 양품 이미지 클래스. YOLO segmentation label 파일은 비어 있음 |
| 1 | `scratch` | 긁힘, 스크래치 |
| 2 | `dent` | 찌그러짐, 눌림 |
| 3 | `stain` | 오염, 얼룩, 액체 자국 |

### Generate

```bash
pip install -r requirements.txt
python scripts/generate_synthetic_can_defects.py --count 500 --clean
```

`--reference-image`를 생략하면 `assets/deepx_can_reference_*.png` 패턴의 레퍼런스를 자동으로 모두 사용합니다.

기본 출력 위치는 `data/synthetic_deepx_can_seg/`입니다.

```text
data/synthetic_deepx_can_seg/
  images/{train,val,test}/
  labels/{train,val,test}/
  masks/{train,val,test}/
  annotations/coco_{train,val,test}.json
  dataset.yaml
  metadata.csv
  preview_grid.jpg
```

`labels/{split}/*.txt`는 Ultralytics YOLO segmentation 포맷입니다.

```text
<class-id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

좌표는 이미지 크기 기준으로 `0..1` 정규화된 폴리곤입니다. 불량이 2개 또는 3개 겹치는 이미지도 생성될 수 있으며, 각 결함 instance는 별도 라벨 행으로 저장됩니다.

YOLO26 segmentation 학습에는 `dataset.yaml`을 사용하면 됩니다.

```bash
yolo segment train data=data/synthetic_deepx_can_seg/dataset.yaml model=yolo26n-seg.pt imgsz=1280 epochs=100
```

옵션 예시:

```bash
python scripts/generate_synthetic_can_defects.py \
  --out data/synthetic_deepx_can_seg \
  --reference-image \
    assets/deepx_can_reference_01.png \
    assets/deepx_can_reference_02.png \
    assets/deepx_can_reference_03.png \
    assets/deepx_can_reference_04.png \
  --count 1000 \
  --normal-ratio 0.15 \
  --max-defects 3 \
  --overlap-ratio 0.35 \
  --seed 42 \
  --clean
```
