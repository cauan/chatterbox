# Quick Start: Chatterbox TTS with MLX on Apple Silicon

Get started with Chatterbox TTS optimized for your Mac with Apple Silicon in under 5 minutes!

## 🚀 One-Command Installation

```bash
bash setup_mlx.sh
```

That's it! The script will:
- ✓ Check your system compatibility
- ✓ Install MLX and all dependencies
- ✓ Install Chatterbox TTS
- ✓ Verify the installation

## 📝 Manual Installation (Alternative)

If you prefer manual installation:

```bash
# 1. Install MLX dependencies
pip install -r requirements-mlx.txt

# 2. Install Chatterbox
pip install -e .

# 3. Verify installation
python -c "import mlx.core as mx; from chatterbox.tts import ChatterboxTTS; print('✓ Ready!')"
```

## 🎯 Your First TTS Generation

### Option 1: Run the Example Script

```bash
python example_mlx.py
```

This will generate multiple example audio files demonstrating all features.

### Option 2: Write Your Own Code

```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Load model (MLX optimizations applied automatically on Apple Silicon)
model = ChatterboxTTS.from_pretrained(device="cpu")

# Generate speech
text = "Hello! This is Chatterbox TTS running on Apple Silicon with MLX optimization."
wav = model.generate(text)

# Save audio
ta.save("my_first_tts.wav", wav, model.sr)
print("✓ Audio saved to my_first_tts.wav")
```

Run it:
```bash
python your_script.py
```

## 🌍 Multilingual Example

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torchaudio as ta

model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")

# Generate in different languages
examples = {
    "en": "Hello, how are you today?",
    "es": "¡Hola! ¿Cómo estás?",
    "fr": "Bonjour! Comment allez-vous?",
    "ja": "こんにちは、元気ですか？",
    "zh": "你好，最近怎么样？",
}

for lang, text in examples.items():
    wav = model.generate(text, language_id=lang)
    ta.save(f"output_{lang}.wav", wav, model.sr)
    print(f"✓ Generated {lang}")
```

## 🎤 Voice Cloning Example

```python
from chatterbox.tts import ChatterboxTTS
import torchaudio as ta

model = ChatterboxTTS.from_pretrained(device="cpu")

text = "This is my voice being cloned by Chatterbox TTS!"
audio_prompt = "path/to/your/voice_sample.wav"  # 3-10 seconds of speech

wav = model.generate(
    text,
    audio_prompt_path=audio_prompt,
    exaggeration=0.5,  # Emotion intensity (0.0-2.0)
    cfg_weight=0.5     # Adherence to prompt (0.0-1.0)
)

ta.save("voice_cloned.wav", wav, model.sr)
```

## 🎭 Expressive Speech

```python
from chatterbox.tts import ChatterboxTTS
import torchaudio as ta

model = ChatterboxTTS.from_pretrained(device="cpu")

text = "I can't believe we actually won! This is amazing!"

# Generate with high exaggeration for dramatic effect
wav = model.generate(
    text,
    exaggeration=0.8,   # Higher = more expressive
    cfg_weight=0.3      # Lower for dramatic speech
)

ta.save("expressive.wav", wav, model.sr)
```

## ⚙️ Performance Tuning

### Reduce Memory Usage

```python
import mlx.core as mx

# Set memory limit (e.g., 4GB)
mx.set_memory_limit(4 * 1024 * 1024 * 1024)

# Disable cache
mx.set_cache_limit(0)

# Now load and use the model
from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained(device="cpu")
```

### Batch Processing

```python
from chatterbox.tts import ChatterboxTTS
import torchaudio as ta

model = ChatterboxTTS.from_pretrained(device="cpu")

texts = [
    "First sentence to generate.",
    "Second sentence to generate.",
    "Third sentence to generate.",
]

for i, text in enumerate(texts):
    wav = model.generate(text)
    ta.save(f"batch_{i}.wav", wav, model.sr)
    print(f"✓ Generated {i+1}/{len(texts)}")
```

## 📊 Benchmark Your System

```python
import time
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cpu")
text = "This is a benchmark test."

# Warmup
model.generate(text)

# Benchmark
start = time.time()
for _ in range(5):
    wav = model.generate(text)
elapsed = time.time() - start

print(f"Average time per generation: {elapsed/5:.2f}s")
```

## 🔧 Troubleshooting

### "MLX not found"
```bash
pip install -r requirements-mlx.txt
```

### "Out of memory"
```python
import mlx.core as mx
mx.set_memory_limit(4 * 1024 * 1024 * 1024)  # Limit to 4GB
```

### Slow first generation?
The first generation downloads model weights (~2GB). Subsequent runs are much faster as weights are cached locally.

## 📚 Learn More

- **Full MLX Guide**: [README_MLX.md](README_MLX.md)
- **Main Documentation**: [README.md](README.md)
- **Discord Community**: https://discord.gg/rJq9cRJBJ6
- **Report Issues**: https://github.com/resemble-ai/chatterbox/issues

## 🎯 Common Use Cases

### AI Voice Agents
```python
# Real-time voice response for chatbots
response_text = get_chatbot_response(user_input)
wav = model.generate(response_text)
play_audio(wav)  # Your audio playback function
```

### Content Creation
```python
# Generate narration for videos
script = load_video_script()
for i, line in enumerate(script):
    wav = model.generate(line)
    ta.save(f"narration_{i:03d}.wav", wav, model.sr)
```

### Game Development
```python
# Generate NPC dialogue
npc_lines = ["Welcome, traveler!", "What brings you here?"]
for i, line in enumerate(npc_lines):
    wav = model.generate(line, exaggeration=0.7)
    ta.save(f"npc_line_{i}.wav", wav, model.sr)
```

### Accessibility Tools
```python
# Screen reader or text-to-speech for visually impaired
text = extract_text_from_screen()
wav = model.generate(text)
play_audio_stream(wav)
```

## 💡 Pro Tips

1. **First letter capitalization**: The model works best when the first letter is capitalized
2. **Punctuation**: Use proper punctuation (., !, ?) for better prosody
3. **Audio prompts**: Use 3-10 seconds of clean speech for best voice cloning
4. **Language matching**: Match the audio prompt language to the target language for best results
5. **Memory**: Close other apps when generating to free up RAM on lower-end Macs

## 🚀 Next Steps

Now that you're up and running:

1. Experiment with different `exaggeration` values (0.0 to 2.0)
2. Try voice cloning with your own voice samples
3. Generate audio in all 23 supported languages
4. Join our [Discord](https://discord.gg/rJq9cRJBJ6) to share your creations

Happy generating! 🎉
