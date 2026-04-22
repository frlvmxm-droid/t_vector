# Quickstart: запуск web-UI на Linux через PuTTY (для Windows-пользователей)

Пошаговый гайд: как с рабочего компьютера на Windows через **PuTTY**
подключиться к Linux-машине, поднять Voilà-дашборд и открыть его в
локальном браузере.

Если вы администратор JupyterHub и хотите раздать дашборд всем
пользователям хаба — смотрите [`JUPYTERHUB_UI.md`](JUPYTERHUB_UI.md).

---

## Что понадобится (один раз)

- **PuTTY** — https://www.putty.org/
- SSH-доступ к Linux-машине: логин, пароль или SSH-ключ
- Браузер на Windows (Chrome / Edge / Firefox)

---

## Вариант A — обычный Linux-сервер (без JupyterHub)

### Шаг 1. PuTTY: подключение + проброс порта

1. Запустите PuTTY.
2. **Session** → `Host Name: <логин>@<ip-или-hostname>`, `Port: 22`.
3. Слева: **Connection → SSH → Tunnels**.
4. Заполните:
   - `Source port`: `8866`
   - `Destination`: `127.0.0.1:8866`
   - Оставьте `Local` + `Auto`
   - Нажмите **Add** — в списке появится `L8866 127.0.0.1:8866`.
5. Вернитесь на **Session** → введите имя в `Saved Sessions` → **Save**
   (чтобы не настраивать каждый раз).
6. **Open** → логиньтесь.

### Шаг 2. В терминале PuTTY: установка (один раз)

```bash
# 1) Клонировать проект
git clone <URL-вашего-репо> t_vector
cd t_vector

# 2) Установить uv (менеджер пакетов, ~10 сек)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env    # или перелогиньтесь

# 3) Установить зависимости — ML + UI extras (~2–5 мин)
uv sync --frozen --extra ml --extra ui
```

### Шаг 3. В терминале PuTTY: запуск Voilà

```bash
cd ~/t_vector   # если не там
PYTHONPATH=. uv run voila notebooks/ui.ipynb \
    --port=8866 --no-browser --Voila.ip=127.0.0.1
```

Увидите:
```
[Voila] Voilà is running at:
http://127.0.0.1:8866/
```

**Не закрывайте PuTTY** — пока он открыт, работает и сервер, и туннель.

### Шаг 4. В браузере на Windows

Откройте: **http://localhost:8866/**

Первый рендер занимает ~30–60 секунд (Voilà поднимает Python-кернел и
импортирует sklearn/ipywidgets) — экран белый, это нормально. Потом
увидите тёмно-бирюзовый sidebar и три вкладки.

### Шаг 5. Остановить

- В PuTTY: **Ctrl+C** — останавливает Voilà.
- Закрыть PuTTY — закрывает туннель.

---

## Вариант B — JupyterHub (уже развёрнут админом)

Если у вас **уже есть** JupyterHub по адресу типа `https://hub.company.com/`,
и админ настроил `default_url=/voila/render/notebooks/ui.ipynb` — всё проще:

1. В браузере зайдите на `https://hub.company.com/`.
2. Залогиньтесь (LDAP / OAuth / локальный — как у вас настроено).
3. Вы **сразу попадёте в дашборд** — PuTTY не нужен.

Если админ **не** перенаправил spawner:

1. Залогиньтесь в Hub обычно → откроется JupyterLab.
2. Откройте терминал в Lab (**File → New → Terminal**).
3. Однократно:
   ```bash
   git clone <URL> t_vector
   cd t_vector
   pip install --user ipywidgets voila scikit-learn joblib pandas openpyxl
   ```
4. Запустите Voilà через Jupyter-proxy:
   ```bash
   cd ~/t_vector
   PYTHONPATH=. voila notebooks/ui.ipynb --port=8866 --no-browser \
       --Voila.ip=127.0.0.1
   ```
5. В браузере откройте (замените `<you>` на ваш логин):
   ```
   https://hub.company.com/user/<you>/proxy/8866/
   ```
   (нужен пакет `jupyter-server-proxy`, который обычно уже установлен
   на Hub).

---

## Вариант C — JupyterHub без server-proxy (fallback на PuTTY-туннель)

Если `jupyter-server-proxy` не стоит — гибрид:

1. В JupyterLab → терминал → запустите Voilà как в Варианте B шаг 4.
2. В PuTTY откройте **второе** SSH-соединение к этому же хосту с
   пробросом `L8866 → 127.0.0.1:8866` (как в Варианте A шаг 1).
3. В браузере: **http://localhost:8866/**.

---

## Что делать в UI

После открытия увидите три вкладки слева:

| Вкладка | Что делает | Что загружать |
|---|---|---|
| 📚 **Обучение** | Обучает TF-IDF + LinearSVC | XLSX/CSV с колонками `text` + `label` |
| 🎯 **Классификация** | Применяет модель | `.joblib` + XLSX/CSV |
| 🧩 **Кластеризация** | KMeans / HDBSCAN / LDA / SBERT | Один/несколько XLSX/CSV с текстами |

**Для больших файлов (>50 МБ)** — не загружайте через браузер. Положите
файл на Linux-хост (например, `/home/<you>/data/big.xlsx`) и в UI
заполните поле **«Путь:»** этим абсолютным путём.

Значения виджетов (K, алгоритм, пороги, колонки, …) сохраняются
автоматически в `~/.classification_tool/last_session.json` и
восстанавливаются при следующем открытии вкладки.

---

## Частые проблемы

| Симптом | Решение |
|---|---|
| В браузере `ERR_CONNECTION_REFUSED` | Туннель не поднят. PuTTY → Tunnels — есть ли `L8866`? Или Voilà упал в терминале. |
| Blank page / бесконечная загрузка | Подождите 60 сек — первый кернел медленно стартует. Если больше — смотрите stderr Voilà в PuTTY. |
| `ModuleNotFoundError: ui_widgets` | Запускаете не из корня репо. `cd ~/t_vector` перед `voila`, обязательно `PYTHONPATH=.`. |
| `Address already in use` на 8866 | Занято прошлым запуском. `pkill -f voila` или `--port=8867`. |
| `ImportError: ipywidgets` | `pip install ipywidgets voila` (или `uv sync --extra ui`). |
| Большие файлы виснут на upload | Не грузите через браузер — используйте «Путь:» на сервере. |
| `UNTRUSTED_MODEL_PATH` при Apply | В UI появится prompt с SHA-256 — подтвердите «Да». Доверенные хеши: `~/.classification_tool/trusted_models.json`. |

---

## Что дальше

- Admin-гайд по JupyterHub: [`JUPYTERHUB_UI.md`](JUPYTERHUB_UI.md)
- Docker / headless CLI: [`DEPLOY.md`](DEPLOY.md)
- Архитектура и developer guide: [`../CLAUDE.md`](../CLAUDE.md)
