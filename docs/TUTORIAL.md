# Tutorial AudioLayer Studio

**🇮🇩 Bahasa Indonesia** · [🇬🇧 English](TUTORIAL.en.md)

Panduan langkah demi langkah dari nol sampai bisa membersihkan audio video AI — lewat aplikasi web
di komputer sendiri **dan** lewat Claude.

---

## Tanpa install: pakai langsung di Claude

1. Di aplikasi Claude (web, desktop, atau HP) buka **Settings → Capabilities**, lalu nyalakan
   **Code execution** (akses jaringan minimal mengizinkan *package managers*).
2. Tambahkan skill: **Settings → Capabilities → Skills → Upload skill**, pilih ZIP dari folder
   [`skills/audiolayer`](../skills/audiolayer) (atau simpan dari kartu skill yang ditawarkan Claude).
3. Buka chat, **upload klip-mu**, lalu tulis misalnya *"bersihkan audio video ini"*.

Claude akan memasang mesin AudioLayer versi ringan (tanpa PyTorch) di sandbox miliknya sendiri
(pertama kali 1–2 menit), memberi tahu isi audionya (musik, dialog, efek, noise, suara ngoceh…),
**menanyakan mana yang mau dihapus**, lalu memberikan file video/audio bersih untuk diunduh.
Tidak ada yang perlu di-install di komputermu. Paling cocok untuk klip pendek (batas upload aplikasi berlaku).

---

## ⚡ Cara tercepat di komputer sendiri (3 langkah)

1. **Install bahan dasar**: Python 3.11+, Git dan FFmpeg ([langkah 2](#2-install-bahan-dasar)).
   Installer di bawah bisa memasang Python/FFmpeg otomatis lewat Homebrew (Mac), apt (Ubuntu) atau winget (Windows).
2. **Download & jalankan installer**:
   ```bash
   git clone https://github.com/zallxz/AudioLayer-Studio.git
   cd AudioLayer-Studio
   ./install.sh                                          # macOS / Linux
   ```
   ```powershell
   powershell -ExecutionPolicy Bypass -File install.ps1  # Windows
   ```
   Installer otomatis: membuat virtual environment, memasang PyTorch + AudioLayer, mengunduh model AI
   (± 550 MB), membuat tombol peluncur, dan **menyambungkan ke Claude Desktop** (+ Claude Code kalau ada).
   Node.js tidak wajib — tampilan web sudah ikut terpasang.
3. **Pakai**:
   - **Aplikasi web** → klik dua kali `AudioLayer.command` (Mac/Linux) atau `AudioLayer.bat` (Windows).
   - **Satu perintah** → `audiolayer clean video.mp4 --preset remove_music --loudness youtube --video`
   - **Claude** → taruh video di folder `AudioLayer/inputs` (di folder home kamu), restart Claude Desktop,
     lalu ketik misalnya *"bersihkan musik latar di scene1.mp4"*.

Cek kapan saja apakah semuanya beres: `audiolayer doctor`.

Langkah-langkah di bawah adalah versi manual/detail dari proses yang sama.

---

## Daftar isi

1. [Apa yang dibutuhkan](#1-apa-yang-dibutuhkan)
2. [Install bahan dasar (Python, FFmpeg, Node.js, Git)](#2-install-bahan-dasar)
3. [Download & install AudioLayer](#3-download--install-audiolayer)
4. [Download model AI](#4-download-model-ai-sekali-saja)
5. [Pakai lewat aplikasi web](#5-pakai-lewat-aplikasi-web)
6. [Pakai lewat Claude Desktop](#6-pakai-lewat-claude-desktop)
7. [Pakai lewat Claude Code](#7-pakai-lewat-claude-code-terminal)
8. [Bisa dipakai di Claude versi web (claude.ai)?](#8-bisa-dipakai-di-claudeai-versi-web) — bisa, lihat Tanpa install
9. [Resep cepat untuk masalah umum video AI](#9-resep-cepat)
   · [Perintah `audiolayer` (CLI)](#perintah-audiolayer-cli) · [Skill Claude](#skill-claude)
10. [Kalau ada masalah](#10-kalau-ada-masalah)

---

## 1. Apa yang dibutuhkan

| | Minimal | Disarankan |
|---|---|---|
| OS | Windows 10/11, macOS 12+, Linux | — |
| RAM | 8 GB | 16 GB |
| Disk kosong | ± 4 GB (aplikasi + model AI) | lebih kalau videonya banyak |
| GPU | tidak wajib | NVIDIA (CUDA) → proses jauh lebih cepat |
| Internet | untuk install & download model pertama kali | setelah itu bisa offline |

Kecepatan di CPU laptop biasa: kira-kira **1,5× durasi video** (video 20 detik ≈ 30 detik proses).
Semua proses jalan di komputer kamu sendiri — video tidak di-upload ke mana pun.

---

## 2. Install bahan dasar

### Windows

1. **Python 3.11 atau lebih baru** — download dari <https://www.python.org/downloads/>.
   Saat install, **centang "Add python.exe to PATH"**.
2. **FFmpeg** — buka *PowerShell* lalu jalankan:
   ```powershell
   winget install Gyan.FFmpeg
   ```
3. **Node.js 20+** (untuk membangun tampilan web): `winget install OpenJS.NodeJS.LTS`
4. **Git**: `winget install Git.Git`
5. **Tutup lalu buka lagi** PowerShell, kemudian cek:
   ```powershell
   python --version
   ffmpeg -version
   node --version
   git --version
   ```

### macOS

1. Install Homebrew (kalau belum): ikuti <https://brew.sh>
2. Lalu di Terminal:
   ```bash
   brew install python@3.12 ffmpeg node git
   ```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg libsndfile1 nodejs npm git
```

---

## 3. Download & install AudioLayer

Buka Terminal (macOS/Linux) atau PowerShell (Windows), lalu:

```bash
git clone https://github.com/zallxz/AudioLayer-Studio.git
cd AudioLayer-Studio
```

### 3a. Buat "ruang kerja" Python (virtual environment)

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
> Kalau Windows menolak menjalankan script: jalankan sekali
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, lalu ulangi.

Kalau berhasil, di awal baris terminal muncul `(.venv)`. **Setiap kali** mau memakai AudioLayer di
terminal baru, aktifkan lagi dengan perintah `activate` di atas.

### 3b. Install PyTorch (mesin AI)

Pilih **salah satu**:

```bash
# Laptop/PC tanpa GPU NVIDIA (Windows / Linux):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# macOS (Apple Silicon / Intel):
pip install torch

# PC dengan GPU NVIDIA: lihat perintah yang tepat di https://pytorch.org/get-started/locally/
```

### 3c. Install AudioLayer

```bash
pip install -e ".[ml]"
```

### 3d. Bangun tampilan web (opsional)

Tampilan web siap pakai sudah ikut terpasang. Langkah ini hanya perlu kalau kamu mengubah kode UI:

```bash
cd frontend
npm ci
npm run build
cd ..
```

---

## 4. Download model AI (sekali saja)

```bash
python scripts/download_models.py
```

Ini mengunduh ± 550 MB: PANNs (pendeteksi suara), Demucs (pemisah suara), Whisper (transkripsi).
Kalau langkah ini dilewati, model tetap terunduh otomatis saat analisis pertama.

Cek status model:
```bash
python scripts/download_models.py --list
```

**Mau coba tanpa video sendiri?** Buat video contoh 20 detik (ucapan + musik + efek + noise):
```bash
python scripts/make_demo_media.py demo
```
Hasilnya: `demo/demo_ai_video.mp4`.

---

## 5. Pakai lewat aplikasi web

### Menjalankan

```bash
audiolayer-api
```

(atau `audiolayer web`, yang juga langsung membuka browser). Biarkan terminal itu terbuka, lalu buka
browser ke **<http://127.0.0.1:8000>**.
Untuk menghentikan: tekan `Ctrl + C` di terminal.

### Alur kerja

1. **Upload** — klik *Upload media* atau seret file (MP4, MOV, MKV, WebM, WAV, MP3, AAC, FLAC) ke halaman.
   Analisis langsung berjalan; progresnya terlihat di bar atas (bisa di-*Cancel*).
2. **Lihat hasil** — timeline menampilkan waveform dan 5 jalur warna:
   **Dialogue** (biru), **BGM** (ungu), **SFX** (oranye), **Music** (pink), **Noise** (abu-abu).
   - Kotak makin pekat = makin yakin. Kotak bergaris = *uncertain* (AI ragu, cek manual).
   - Tanda **?!** merah di dialog = suara "ngoceh" / tidak jelas (gibberish).
   - Panel kiri bawah: **Detected problems** (hiss, hum, clipping, loudness, dll.) beserta tombol perbaikan.
3. **Dengarkan** — tekan ▶ atau `Spasi`. Tombol kategori di bar bawah (Dialogue, BGM, …) mematikan/menyalakan
   kategori itu; yang kamu dengar = yang akan di-export. **A/B original** untuk membandingkan dengan audio asli.
4. **Perbaiki label** — klik kotak di timeline → panel kanan menampilkan detailnya. Kalau salah kategori
   (misal lagu dikira BGM), klik kategori yang benar. Bisa juga:
   - **Isolate**: dengar segmen itu saja.
   - **Remove segment**: buang hanya bagian itu.
   - **Remove all …**: buang seluruh kategori.
   - Ubah *Start/End* untuk merapikan batas waktu.
   - Salah klik? Tombol **↶ / ↷** (undo/redo) di atas.
5. **Quick fixes** (panel kiri) — satu klik untuk kasus umum: *Remove all music*, *Dialogue only*,
   *Music & Effects (M&E)*, *Remove noise*, *Remove sound effects*, *Mute gibberish dialogue*.
6. **Export** — tombol kuning *Export*:
   - Pilih file: cleaned mix, stem per kategori, audio asli, **video dengan audio bersih**, metadata.
   - Format: WAV 48 kHz 24-bit / FLAC / MP3.
   - **Loudness**: YouTube/TikTok −14 LUFS, Podcast −16, Broadcast −23 (bikin volume antar klip sama).
   - Klik *Export* → *Download ZIP*.

Semua file stem panjangnya sama dan mulai dari detik 0 — taruh semuanya di awal timeline
Premiere/DaVinci/CapCut/Audacity dan semuanya langsung sinkron. File `*_audacity_labels.txt` bisa
di-import di Audacity (*File → Import → Labels*) untuk melihat penanda segmen.

File hasil juga tersimpan di folder `AudioLayer/workspace/projects/<id>/exports/` di folder home kamu.

---

## 6. Pakai lewat Claude Desktop

**Ya, AudioLayer bisa dipakai langsung dari Claude.** Claude Desktop menjalankan AudioLayer di
komputermu lewat MCP, jadi kamu cukup menyuruh Claude dengan bahasa biasa.

### Cara otomatis (disarankan)

```bash
audiolayer setup-claude --input-dir "~/Documents/Video AI"          # tambahkan --code untuk Claude Code juga
```
Perintah ini menulis konfigurasi Claude Desktop untukmu (file lama di-backup dulu), lalu tinggal
restart Claude Desktop dan lompat ke [Langkah 5](#langkah-5--suruh-claude). `install.sh`/`install.ps1`
sudah menjalankannya dengan folder `~/AudioLayer/inputs`.

### Cara manual

#### Langkah 1 — Cari lokasi program `audiolayer-mcp`

Dengan venv aktif:

macOS / Linux:
```bash
which audiolayer-mcp
# contoh hasil: /Users/nama/AudioLayer-Studio/.venv/bin/audiolayer-mcp
```
Windows:
```powershell
where.exe audiolayer-mcp
# contoh hasil: C:\Users\nama\AudioLayer-Studio\.venv\Scripts\audiolayer-mcp.exe
```

#### Langkah 2 — Tentukan folder video

Claude hanya boleh membaca video dari folder yang kamu izinkan (demi keamanan). Misalnya buat folder
`Video AI` di Documents dan simpan klip-klipmu di sana.

#### Langkah 3 — Edit konfigurasi Claude Desktop

Buka Claude Desktop → **Settings → Developer → Edit Config**. File yang terbuka:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Isi (sesuaikan path-nya):

**macOS / Linux**
```json
{
  "mcpServers": {
    "audiolayer": {
      "command": "/Users/nama/AudioLayer-Studio/.venv/bin/audiolayer-mcp",
      "env": {
        "AUDIOLAYER_WORKSPACE": "/Users/nama/AudioLayer/workspace",
        "AUDIOLAYER_INPUT_DIRS": "/Users/nama/Documents/Video AI",
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
      }
    }
  }
}
```
> `PATH` perlu supaya Claude menemukan FFmpeg dari Homebrew.

**Windows** (garis miring terbalik harus ditulis dobel `\\`)
```json
{
  "mcpServers": {
    "audiolayer": {
      "command": "C:\\Users\\nama\\AudioLayer-Studio\\.venv\\Scripts\\audiolayer-mcp.exe",
      "env": {
        "AUDIOLAYER_WORKSPACE": "C:\\Users\\nama\\AudioLayer\\workspace",
        "AUDIOLAYER_INPUT_DIRS": "C:\\Users\\nama\\Documents\\Video AI"
      }
    }
  }
}
```

Kalau file config sudah berisi server MCP lain, tambahkan blok `"audiolayer": {...}` di dalam
`"mcpServers"` yang sudah ada (pisahkan dengan koma). Lebih dari satu folder video:
pisahkan dengan `:` (macOS/Linux) atau `;` (Windows).

#### Langkah 4 — Restart Claude Desktop

Tutup Claude Desktop sepenuhnya (termasuk dari tray/menu bar), lalu buka lagi. Di kolom chat,
menu tools/konektor sekarang menampilkan **audiolayer**.

### Langkah 5 — Suruh Claude

Contoh perintah:

- *"Analisis `scene1.mp4` di folder Video AI pakai AudioLayer. Ada suara apa saja?"*
- *"Hapus musik latarnya tapi dialog dan efek suaranya tetap, lalu export video dengan audio bersih
  untuk YouTube."*
- *"Ada dialog ngoceh/gibberish nggak? Kalau ada, bisukan bagian itu saja."*
- *"Bikinkan versi Music & Effects (tanpa suara orang) buat aku dubbing ulang."*
- *"Cek kualitas audionya: ada hum, hiss, clipping atau volume terlalu kecil?"*
- *"Segmen musik di detik 12 itu sebenarnya lagu, bukan BGM — perbaiki labelnya."*
- *"Export semua stem dalam format WAV."*

Claude akan menjalankan analisis, menunggu sampai selesai, membacakan hasilnya (termasuk batasan /
bagian yang kurang yakin), lalu memberi lokasi file hasil di komputermu.

**Bonus:** project yang dibuat lewat Claude juga muncul di aplikasi web (selama `AUDIOLAYER_WORKSPACE`
sama) — jadi bisa minta Claude menganalisis, lalu kamu cek dan dengarkan di timeline.

### Tools yang tersedia untuk Claude

| Tool | Fungsi |
|---|---|
| `analyze_media` | Mulai analisis file video/audio |
| `get_processing_status` / `cancel_processing` | Cek progres / batalkan |
| `get_analysis`, `get_quality_report` | Ringkasan hasil & masalah audio |
| `list_audio_segments` | Daftar segmen (waktu, kategori, keyakinan, transkrip) |
| `update_segment_category` | Ganti label / waktu / bisukan segmen |
| `remove_audio_category` | Pilih kategori yang dibuang atau pakai preset |
| `separate_audio_category` | Ambil satu kategori saja sebagai file |
| `export_audio_stems`, `export_cleaned_audio` | Export stem / audio bersih (+ video) |
| `list_projects`, `get_system_info` | Info project & model |

---

## 7. Pakai lewat Claude Code (terminal)

Cara termudah (venv aktif):

```bash
audiolayer setup-claude --input-dir "$HOME/Documents/Video AI" --code
```

Atau manual:

```bash
claude mcp add audiolayer --scope user \
  -e AUDIOLAYER_INPUT_DIRS="$HOME/Documents/Video AI" \
  -e AUDIOLAYER_WORKSPACE="$HOME/AudioLayer/workspace" \
  -- "$(which audiolayer-mcp)"
```

Lalu di Claude Code: *"pakai audiolayer untuk menghapus musik dari ~/Documents/Video AI/scene1.mp4"*.

---

## 8. Bisa dipakai di claude.ai (versi web)?

**Bisa — lewat skill (mode cloud).** Upload klip di chat, Claude memprosesnya di sandbox miliknya
sendiri (lihat [Tanpa install](#tanpa-install-pakai-langsung-di-claude) di bagian atas). Yang tidak
bisa: claude.ai menjangkau program yang berjalan di laptopmu (`localhost`). Untuk file di komputermu:

- **Paling mudah:** pakai **Claude Desktop** (langkah 6) — gratis diunduh dari <https://claude.ai/download>.
- **Lanjutan:** jalankan AudioLayer di server sendiri dengan alamat HTTPS publik, lalu tambahkan
  sebagai *custom connector* di claude.ai (kalau paket/organisasimu mengizinkan). Panduan:
  [docs/MCP.md → Remote deployment](MCP.md#remote-deployment).

---

## 9. Resep cepat

| Masalah di video AI | Yang dilakukan |
|---|---|
| Ada musik yang tidak diminta | Quick fix **Remove all music** → Export |
| Karakter bicara bahasa aneh / ngoceh | **Mute gibberish dialogue**, atau **Music & Effects** untuk buang semua dialog |
| Mau ganti suara AI dengan voice-over sendiri | **Music & Effects (M&E)** → export, lalu tambahkan suara baru di editor |
| Butuh suara dialog saja (untuk subtitle / dubbing) | **Dialogue only**, export stem Dialogue + file `.srt` |
| Desis (hiss) / dengung (hum) | **Remove noise** |
| Volume beda-beda antar klip yang disambung | Export semua klip dengan loudness **YouTube −14 LUFS** + edge fade 5 ms |
| Suara terdengar "teredam" | Dilaporkan sebagai *Band-limited audio* — tidak bisa diperbaiki dengan pemisahan; generate ulang atau pakai tool bandwidth-extension |
| Video tanpa suara | Dilaporkan sebagai *No audio track / Silent audio* — generate ulang dengan audio aktif |

**Yang perlu diketahui (jujur):** pemisahan suara orang dari bagian lain bekerja cukup baik; tetapi
musik dan efek suara yang **berbunyi bersamaan** belum bisa dipisah bersih (ditandai "split is
estimated"). Efek yang mirip pukulan drum kadang terdeteksi sebagai musik. Selalu dengarkan hasilnya
dengan tombol **A/B original** sebelum export. Detail: [docs/MODELS.md](MODELS.md).

### Perintah `audiolayer` (CLI)

| Perintah | Fungsi |
|---|---|
| `audiolayer web` | Buka aplikasi web di browser |
| `audiolayer analyze video.mp4` | Lihat isi audio (segmen, transkrip, masalah) |
| `audiolayer clean video.mp4 --preset remove_music --video` | Hapus musik, simpan video bersih |
| `audiolayer clean video.mp4 --remove sfx,noise --stems --zip` | Hapus SFX & noise, simpan semua stem + ZIP |
| `audiolayer clean video.mp4 --mute-gibberish --loudness tiktok` | Bisukan dialog ngoceh, volume standar TikTok |
| `audiolayer setup-claude --input-dir <folder>` | Sambungkan ke Claude Desktop (`--code` untuk Claude Code) |
| `audiolayer models download` | Unduh model AI |
| `audiolayer doctor` | Cek instalasi |

Hasil `clean` tersimpan di folder `<nama-video>_audiolayer` di sebelah video aslinya.
Preset: `remove_music`, `dialogue_only`, `music_and_effects`, `remove_noise`, `remove_sfx`, `remove_gibberish`.

### Skill Claude

Repo ini menyertakan skill Claude di [`skills/audiolayer/SKILL.md`](../skills/audiolayer/SKILL.md).
Skill membuat Claude otomatis tahu kapan dan bagaimana memakai AudioLayer: menerjemahkan permintaan
("hapus musiknya", "suaranya ngoceh", "buat dubbing") jadi preset yang tepat, menjalankan analisis,
menunggu hasil, dan melaporkan keterbatasan dengan jujur.

- **Claude (aplikasi/web)**: *Settings → Capabilities → Skills → Upload skill*, lalu pilih file ZIP
  berisi folder `audiolayer/` (atau simpan dari kartu skill yang ditawarkan Claude).
- **Claude Code**: salin folder ke `~/.claude/skills/audiolayer/` (semua project) atau
  `.claude/skills/audiolayer/` (satu project).

Skill paling berguna bersama MCP server (Claude Desktop) atau CLI (Claude Code). Tanpa keduanya,
skill akan memandu cara install-nya dulu.

---

## 10. Kalau ada masalah

| Gejala | Solusi |
|---|---|
| Tidak yakin apa yang salah | Jalankan `audiolayer doctor` — semua bagian dicek satu per satu |
| `audiolayer` / `audiolayer-api` / `audiolayer-mcp` *not found* | Venv belum aktif — jalankan lagi perintah `activate` (langkah 3a), atau pakai peluncur `AudioLayer.command`/`.bat` |
| `FFmpeg not found on PATH` | Install FFmpeg (langkah 2), tutup-buka terminal. Untuk Claude Desktop di macOS tambahkan `PATH` di config (langkah 6) |
| Claude Desktop tidak menampilkan audiolayer | Cek path `command` benar & absolut; JSON valid (koma, `\\` di Windows); restart penuh. Log: macOS `~/Library/Logs/Claude/mcp-server-audiolayer.log`, Windows `%APPDATA%\Claude\logs` |
| Claude bilang `Path is outside the allowed input folders` | Pindahkan video ke folder di `AUDIOLAYER_INPUT_DIRS`, atau tambahkan foldernya |
| "No source-separation model available" | PyTorch / `.[ml]` belum terinstall (langkah 3b–3c) |
| Download model gagal | Cek internet, jalankan `python scripts/download_models.py` lagi |
| Halaman web kosong / 404 | Jalankan `audiolayer doctor`; kalau baris *web UI* gagal, install ulang (`pip install -e ".[ml]"`) |
| Tidak ada suara di browser | Klik ▶ sekali (browser memblokir audio sebelum ada klik); pastikan status *analyzed* |
| Claude gagal memasang AudioLayer di sandbox-nya | Nyalakan *Code execution* dan izinkan package managers (PyPI + GitHub) di *Settings → Capabilities* |
| Proses lama / laptop panas | Normal di CPU. Matikan *Transcribe dialogue*, atau set `AUDIOLAYER_WHISPER_MODEL=tiny` |
| Kehabisan memori untuk video panjang | Set `AUDIOLAYER_CHUNK_SECONDS=15` |

Masih bermasalah? Buka *Issue* di <https://github.com/zallxz/AudioLayer-Studio/issues> dan sertakan
pesan error + OS yang dipakai.
