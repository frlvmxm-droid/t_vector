#!/usr/bin/env bash
# =============================================================
# run.sh — запуск BankReasonTrainer на Linux / macOS
#
# Первый запуск:
#   chmod +x run.sh
#   ./run.sh
#
# Последующие:
#   ./run.sh
# =============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo " BankReasonTrainer — Запуск (Linux/macOS)"
echo "============================================================"
echo ""

# ── 1. Определяем ОС и дистрибутив ──────────────────────────────────────────
OS="$(uname -s)"      # Linux | Darwin | ...
ARCH="$(uname -m)"   # x86_64 | arm64 | aarch64 | ...
echo "ОС    : $OS ($ARCH)"

# ── 2. Ищем python3 ──────────────────────────────────────────────────────────
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        # Проверяем что это Python 3, а не Python 2
        ver=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "0")
        if [ "$ver" = "3" ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    echo "[ОШИБКА] Python 3 не найден."
    echo ""
    if [ "$OS" = "Linux" ]; then
        # Определяем дистрибутив
        DISTRO_ID=""
        if [ -f /etc/os-release ]; then
            DISTRO_ID=$(grep "^ID=" /etc/os-release | cut -d= -f2 | tr -d '"' | tr '[:upper:]' '[:lower:]')
        fi
        case "$DISTRO_ID" in
            ubuntu|debian|linuxmint|pop|elementary)
                echo "  Установите: sudo apt-get install python3 python3-pip" ;;
            fedora)
                echo "  Установите: sudo dnf install python3 python3-pip" ;;
            rhel|centos|almalinux|rocky|ol)
                echo "  Установите: sudo dnf install python3 python3-pip" ;;
            arch|manjaro|endeavouros|garuda)
                echo "  Установите: sudo pacman -S python python-pip" ;;
            opensuse*|sles)
                echo "  Установите: sudo zypper install python3 python3-pip" ;;
            *)
                echo "  Для Debian/Ubuntu:  sudo apt-get install python3 python3-pip"
                echo "  Для Fedora/RHEL:    sudo dnf install python3 python3-pip"
                echo "  Для Arch:           sudo pacman -S python python-pip"
                ;;
        esac
    elif [ "$OS" = "Darwin" ]; then
        echo "  Установите через Homebrew: brew install python3"
        echo "  Или скачайте: https://www.python.org/downloads/"
    fi
    echo ""
    exit 1
fi

PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "Python: $PY_VER  ($PYTHON)"

# ── 3. Проверяем версию Python (>= 3.9) ──────────────────────────────────────
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
    echo ""
    echo "[ОШИБКА] Требуется Python 3.9 или новее. Найдена версия $PY_VER."
    echo "         Скачайте: https://www.python.org/downloads/"
    echo ""
    exit 1
fi

# ── 4. Проверяем tkinter (на Linux часто нужен отдельный пакет) ───────────────
if ! "$PYTHON" -c "import tkinter" 2>/dev/null; then
    echo ""
    echo "[ОШИБКА] tkinter не найден — требуется для GUI."
    echo ""

    if [ "$OS" = "Linux" ]; then
        DISTRO_ID=""
        if [ -f /etc/os-release ]; then
            DISTRO_ID=$(grep "^ID=" /etc/os-release | cut -d= -f2 | tr -d '"' | tr '[:upper:]' '[:lower:]')
        fi
        case "$DISTRO_ID" in
            ubuntu|debian|linuxmint|pop|elementary)
                echo "  Установите: sudo apt-get install python3-tk"
                echo ""
                read -r -p "  Установить сейчас? [y/N] " ans
                if [[ "${ans,,}" == "y" ]]; then
                    sudo apt-get install -y python3-tk
                else
                    exit 1
                fi ;;
            fedora)
                echo "  Установите: sudo dnf install python3-tkinter"
                echo ""
                read -r -p "  Установить сейчас? [y/N] " ans
                if [[ "${ans,,}" == "y" ]]; then
                    sudo dnf install -y python3-tkinter
                else
                    exit 1
                fi ;;
            rhel|centos|almalinux|rocky|ol)
                echo "  Установите: sudo dnf install python3-tkinter"
                echo ""
                read -r -p "  Установить сейчас? [y/N] " ans
                if [[ "${ans,,}" == "y" ]]; then
                    sudo dnf install -y python3-tkinter
                else
                    exit 1
                fi ;;
            arch|manjaro|endeavouros|garuda)
                echo "  Установите: sudo pacman -S tk"
                echo ""
                read -r -p "  Установить сейчас? [y/N] " ans
                if [[ "${ans,,}" == "y" ]]; then
                    sudo pacman -S --noconfirm tk
                else
                    exit 1
                fi ;;
            opensuse*|sles)
                echo "  Установите: sudo zypper install python3-tk"
                echo ""
                read -r -p "  Установить сейчас? [y/N] " ans
                if [[ "${ans,,}" == "y" ]]; then
                    sudo zypper install -y python3-tk
                else
                    exit 1
                fi ;;
            *)
                echo "  Для Debian/Ubuntu:  sudo apt-get install python3-tk"
                echo "  Для Fedora/RHEL:    sudo dnf install python3-tkinter"
                echo "  Для Arch:           sudo pacman -S tk"
                ;;
        esac
    elif [ "$OS" = "Darwin" ]; then
        echo "  Рекомендуется Python через Homebrew:"
        echo "    brew install python-tk"
        echo "  Или: https://www.python.org/downloads/"
    fi
    echo ""
    exit 1
fi

echo "tkinter: OK"
echo ""

# ── 5. Передаём управление bootstrap_run.py (он установит pip-зависимости) ───
exec "$PYTHON" "$SCRIPT_DIR/bootstrap_run.py"
