#!/usr/bin/env bash
# Downloads the Kokoro ONNX TTS files used by the pilot into cache/tts and checks SHA-256.
# AudioSeal and Whisper tiny weights download automatically on first run (into cache/).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p cache/tts
base=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
get() { # file sha256
  [ -f "cache/tts/$1" ] || curl -L --fail -o "cache/tts/$1" "$base/$1"
  echo "$2  cache/tts/$1" | shasum -a 256 -c -
}
get kokoro-v1.0.onnx 7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5
get voices-v1.0.bin bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d
