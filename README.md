# Чай — синтез фазового ДОЭ

Нейросеть совместно принимает **4 изображения** и выдаёт один `doe.png`: **PNG, 128×128, 8 бит, Grayscale (L)**. Обученные веса включены в репозиторий. Контейнер работает на CPU; после сборки доступ к сети не нужен. Сборка и запуск без сети проверены на пяти комплектах (Linux/arm64), выходные PNG соответствуют требуемому формату.

## Сборка и запуск Docker

Нужен установленный Docker. Python и библиотеки на компьютере устанавливать не требуется.

```bash
git clone git@github.com:tea-nuri/ai-photonics.git
cd ai-photonics
docker build -t tea-photonics .
```

Положите четыре входных PNG в каталог `input` и создайте каталог результатов:

```text
input/
  1.png  # 24.1 мм
  2.png  # 25.3 мм
  3.png  # 26.5 мм
  4.png  # 27.7 мм
```

```bash
mkdir -p output
docker run --rm --network none \
  --mount type=bind,src="$(pwd)/input",dst=/input,readonly \
  --mount type=bind,src="$(pwd)/output",dst=/output \
  tea-photonics --input-dir /input --output-dir /output
```

Результаты: **`output/doe.png`**, **`recon_1.png` … `recon_4.png`** и **`metrics.json`**. Все пять PNG — 128×128, 8 бит, Grayscale L. Восстановления рассчитываются дифракцией Френеля из сохранённого и повторно прочитанного ДОЭ. JSON содержит SSIM и Pearson каждой из четырёх пар и их средние; метрики рассчитываются по сохранённым восстановленным PNG. Входы должны быть PNG 128×128, 8 бит, одноканальные L. `--input-dir` сортирует ровно четыре PNG по имени. Порядок важен: **24.1, 25.3, 26.5, 27.7 мм**. Для явного порядка замените `--input-dir /input` на:

```bash
--inputs /input/1.png /input/2.png /input/3.png /input/4.png
```

Параметры и справка:

```bash
docker run --rm tea-photonics --help
```

## Несколько тестовых комплектов

В `input` создайте подкаталог для каждого комплекта, в каждом — четыре PNG. Вместо `--input-dir` передайте `--batch-dir`:

```bash
docker run --rm --network none \
  --mount type=bind,src="$(pwd)/input",dst=/input,readonly \
  --mount type=bind,src="$(pwd)/output",dst=/output \
  tea-photonics --batch-dir /input --output-dir /output
```

Например, для `input/set_01/` и `input/set_02/` будут записаны `output/set_01/doe.png` и `output/set_02/doe.png`. Веса загружаются один раз. В stdout выводится JSON с путём результата и временем нейросетевого прохода каждого комплекта; загрузка весов и чтение/запись PNG в это время не входят.

## Модель

Полный граф и все параметры находятся в `weights/model.pt`; другие файлы весов и загрузки из интернета для инференса не нужны. Код сети — в `photonics/`, точка входа — `predict.py`. Все четыре плоскости участвуют в расчёте общей фазы. Используются обученные CNN, локальные и глобальные нейросетевые сообщения, фиксированные оптические FFT-слои. При предсказании выполняется фиксированный нейросетевой граф в `torch.inference_mode()`.

Оптика: длина волны **532 нм**, шаг ДОЭ **10 мкм**, расстояния **24.1 / 25.3 / 26.5 / 27.7 мм**. Код фазы: `phase = pixel / 255 × 2π − π`.

Сохранён вариант с лучшим совместным приростом двух средних метрик относительно предыдущей модели: сначала максимизировался меньший из приростов SSIM и Pearson, затем их сумма. На локальной валидации из **128 комплектов** после 8-битного кодирования ДОЭ: **SSIM 0.427390, Pearson 0.486643**. Это локальная оценка, не скрытые тесты организаторов. Сравнивается амплитуда френелевского восстановления; каждое изображение нормируется на собственный максимум. SSIM: Gaussian 11×11, σ=1.5, data_range=1, population covariance; Pearson — по всему изображению. Метрики усредняются по четырём плоскостям, затем по комплектам.

## Локальный запуск без Docker

Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python predict.py --input-dir input --output-dir output
```

CPU используется по умолчанию, 4 потока. `--device mps` включает Apple Metal, `--device cuda` — доступную CUDA; Docker-образ содержит CPU-версию PyTorch. Параметр `--threads` задаёт число CPU-потоков.
