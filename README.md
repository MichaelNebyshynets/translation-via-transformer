# Машинный перевод EN → RU на трансформере с нуля

Encoder-decoder трансформер (PyTorch), написанный с нуля по мотивам видео Андрея Карпатого «Let's build GPT» и расширенный до задачи перевода: кросс-внимание, маски паддинга, teacher forcing, greedy и beam search декодирование. Направление: английский → русский, корпус Tatoeba.

## Результаты

Оценка на 1000 случайных тестовых парах (seed 0, источник ≤ 48 токенов), один референс, sacrebleu:

| Декодирование | BLEU | ratio (длина гипотез / референсов) |
|---|---|---|
| greedy (beam = 1) | 35.56 | 0.981 |
| beam = 2 | 36.36 | 0.973 |
| beam = 4 | 36.79 | 0.966 |
| beam = 8 | 37.10 | 0.958 |

Лучшая валидационная метрика: val loss 1.316, perplexity 3.7 (без label smoothing).

Примеры (greedy):

| EN | Перевод модели |
|---|---|
| I love you. | Я люблю тебя. |
| Tom is waiting for Mary in Boston. | Том ждёт Мэри в Бостоне. |
| What is destiny? | Что такое судьба? |
| Why did you try to stop Tom? | Почему ты пытался остановить Тома? |

Типичные ошибки: падежное согласование («добраться до границу»), буквальные кальки идиом («Давайте примем наше время» вместо «Давайте не будем торопиться»), редкая лексика.

**Оговорки к цифрам.**
- В Tatoeba у одного английского предложения часто несколько допустимых русских переводов, а референс один, поэтому BLEU занижен относительно реального качества.
- BLEU посчитан на подвыборке (1000 пар) тестовых примеров с источником не длиннее 48 токенов, как и при обучении. Длинные предложения (позиционные эмбеддинги рассчитаны на `max_len = 128`) не оценивались.
- Рост BLEU от beam search невелик, а время декодирования растёт сильно (на M5 MacBook Air примерно 1.3 мин при `beam=1`, 17 мин при `beam=4`, 37 мин при `beam=8` на 1000 предложений).

## Модель

| Параметр | Значение |
|---|---|
| Архитектура | encoder-decoder, Pre-LN, 3 слоя в энкодере и в декодере |
| Размерность `n_embd` | 256 |
| Головы внимания | 4 (по 64) |
| FFN | 4 × n_embd, ReLU |
| Позиционные эмбеддинги | обучаемые, `max_len = 128` |
| Эмбеддинги токенов | отдельные для энкодера и декодера (без weight tying) |
| Словарь | 16000, общий BPE (SentencePiece), служебные: pad=0, unk=1, bos=2, eos=3, byte fallback |
| Параметров | 17 893 248 |

## Данные

Корпус: Tatoeba EN–RU, датасет Hugging Face [`xmarva/tfs-tatoeba-en-ru`](https://huggingface.co/datasets/xmarva/tfs-tatoeba-en-ru) (train 508 093 / validation 10 370 / test 36 385).

Предобработка (`prepare_data.ipynb`):
- нормализация Unicode (NFKC) и пробелов;
- удаление пустых пар, пар с одинаковыми en и ru, пар с неправильным алфавитом (доля кириллицы в ru ≤ 0.5 или в en ≥ 0.1);
- удаление из train пар, чьи английские предложения встречаются в validation/test (утечка данных);
- удаление дубликатов;
- обучение BPE-токенизатора (`tokenizer/tokenizer.ipynb`) и токенизация всех сплитов;
- фильтр по длине на train: источник ≤ 48, цель ≤ 50 токенов (с BOS/EOS).

Итоговые размеры: train 507 684, validation 10 370, test 36 380.

## Обучение

| Параметр | Значение |
|---|---|
| Оптимизатор | AdamW, lr 5e-4, betas (0.9, 0.98), weight decay 0.01 |
| Расписание lr | линейный warmup 1000 шагов, затем косинус до 10% от пика |
| Батч | 128 пар |
| Эпох | 10 (≈ 39 660 шагов) |
| Регуляризация | dropout 0.1, label smoothing 0.1 (только на train), grad clip 1.0 |
| Железо | Google Colab, GPU T4 (≈ 8–10 шагов/с) |

Чекпоинты: `last.pt` сохраняется в конце каждой эпохи, `best.pt` по лучшему val loss. При обрыве сессии `train.py` автоматически продолжает с `last.pt`.

## Структура проекта

```
dataset.py            Dataset и collate_fn (паддинг, tgt_in / tgt_out)
model.py              Head, MultiHead, FeedForward, EncoderBlock, DecoderBlock, Encoder, Decoder, Transformer
generate.py           greedy-декодирование и beam search
train.py              обучение, валидация, чекпоинты, продолжение с last.pt
evaluate.py           BLEU на тестовой выборке (greedy / beam)
check_beam.py         проверки beam search и сравнение значений beam
prepare_data.ipynb    загрузка, очистка и токенизация данных
dataset.ipynb         проверки датасета
model.ipynb           тесты слоёв и модели (маски, утечка будущего, overfit)
tokenizer/            bpe.model, bpe.vocab, tokenizer.ipynb
```

Данные (`data/`) и чекпоинты (`checkpoints/`) в репозиторий не входят.

## Запуск

```bash
pip install -r requirements.txt
```

1. Подготовь данные: выполни `prepare_data.ipynb` и `tokenizer/tokenizer.ipynb`. Появятся `data/clean/`, `data/tokenized/` и `tokenizer/bpe.model`.
2. Обучение:
   ```bash
   python -u train.py
   # или с другим каталогом чекпоинтов (например, на Google Drive в Colab):
   CKPT_DIR=/content/drive/MyDrive/mt/checkpoints python -u train.py
   ```
3. Оценка:
   ```bash
   python evaluate.py --beam 4
   ```
4. Перевод одного предложения:
   ```python
   import torch, sentencepiece as spm
   import model as M
   from generate import generate, beam_search

   sp = spm.SentencePieceProcessor(model_file="tokenizer/bpe.model")
   M.dropout = 0.1
   model = M.Transformer()
   model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu")["model"])
   model.eval()

   ids = sp.encode("Where is the station?")
   print(sp.decode(beam_search(model, ids, beam=4)))
   ```

## Идеи для улучшения

- weight tying (эмбеддинги декодера и выходной слой);
- смешанная точность и больший батч для ускорения обучения;
- более глубокая/широкая модель и более длинное обучение;
- метрики кроме BLEU (chrF) и оценка на нескольких референсах.
