# ACE Studio - Streamlit UI for ACE-Step

A modern, user-friendly Streamlit interface for [ACE-Step v1.5](https://github.com/ace-step/ACE-Step-1.5) music generation.

## Features

- 🎵 **Generate** - Create music from text descriptions
- 🎤 **Cover** - Generate cover versions of songs
- 🎨 **Edit** - Repaint song sections, extract vocals, complete sections
- 📦 **Batch** - Generate up to 8 songs simultaneously
- 💾 **Projects** - Save and organize your music creations
- ⚙️ **Settings** - Configure hardware, models, and storage

3. Open your browser to `http://localhost:8501`

### First Generation

1. Go to **Generate** tab
2. Describe your song (e.g., "Upbeat pop with electric guitars")
3. Adjust duration, BPM, and other settings
4. Click **Generate Song**
5. Wait for generation (first run may take longer to load models)


## Configuration

Edit `config.py` to customize:
- Default generation parameters (duration, BPM, guidance scale)
- UI display options
- Storage locations
- Supported audio formats

## Usage Guide

### Dashboard (🎹 Home)

Shows recent projects and quick-start options. Click on any project to:
- **▶️** - Play the audio
- **✏️** - Edit with advanced tools
- **🗑️** - Delete the project

### Generation Wizard (🎵 Generate)

Create new songs in 3 steps:
1. **Inspiration** - Choose genre/mood or describe your song
2. **Structure** - Set duration, BPM, key, and lyrics
3. **Advanced** - Fine-tune diffusion steps, guidance scale, and more

### Audio Editor (🎛️ Edit)

Edit existing songs:
- **Repaint** - Replace a time section with new generation
- **Cover** - Create cover versions with different vocals/style
- **Extract** - Isolate vocals, drums, or other stems
- **Complete** - Generate missing sections

### Batch Generator (📦 Batch)

Generate multiple songs at once:
1. Write song descriptions in queue
2. Add up to 8 songs
3. Configure batch settings
4. Click **Generate All**

Results are saved as separate projects.

### Settings (⚙️ Settings)

Configure:
- **Hardware** - GPU, CUDA, Flash Attention options
- **Models** - Select DiT and LLM models, backends
- **Storage** - Manage projects, clear cache
- **About** - Links to ACE-Step resources

## Keyboard Shortcuts

- `R` - Refresh current tab
- `S` - Open Settings
- `D` - Go to Dashboard

## Troubleshooting

### "Failed to load DiT handler"
- Ensure ACE-Step is installed in parent directory
- Check PyTorch and CUDA installation
- Run `python -c "import torch; print(torch.cuda.is_available())"` to verify

### Models not found
- Models auto-download on first use
- Check internet connection during first generation
- See Settings > Storage to pre-download models

### Out of Memory (OOM)
- Reduce inference steps in advanced settings
- Enable Model Offload in settings
- Run on GPU with larger VRAM

### Audio quality issues
- Increase inference steps (32-100)
- Increase guidance scale (7.5-10.0)
- Use base model instead of turbo (slower but higher quality)

## Performance Tips

- First generation takes longer (model loading)
- Use batch mode for multiple songs (more efficient)
- Enable Flash Attention if GPU supports it
- Turbo model is faster; base model is higher quality


## Links

- 🌍 [ACE-Step Website](https://ace-step.github.io/)
- 🤗 [HuggingFace Model](https://huggingface.co/ACE-Step/Ace-Step1.5)
---

Made with ❤️ for the music generation community
