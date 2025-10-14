# MiniCPM-V Setup Guide

Quick setup guide for using MiniCPM-V nodes in ComfyUI.

## Quick Start

### 1. Install Dependencies

Run this command in your ComfyUI Python environment:

```bash
pip install transformers>=4.40.0 torch>=2.0.0 decord>=0.6.0 scipy>=1.10.0
```

Or use the requirements file:

```bash
cd ComfyUI/custom_nodes/comfyui_dagthomas
pip install -r requirements.txt
```

### 2. Verify Installation

After restarting ComfyUI, you should see two new nodes:

- **MiniCPM-V Image Understanding** (under `comfyui_dagthomas`)
- **MiniCPM-V Video Understanding** (under `comfyui_dagthomas`)

### 3. First Time Usage

**For Images:**

1. Add a `LoadImage` node
2. Add a `MiniCPM-V Image Understanding` node
3. Connect the image output to the images input
4. Set your question in the node
5. Run!

**For Videos:**

1. Add a `MiniCPM-V Video Understanding` node
2. Enter the full path to your video file
3. Set your question
4. Adjust fps (5 is a good default)
5. Run!

## Important Notes

### First Run

- **First time will be slow**: The model (~17GB) needs to download from Hugging Face
- **Requires internet**: For initial model download
- **Disk space**: Ensure you have ~20GB free space
- **Model location**: `~/.cache/huggingface/hub/` (or `C:\Users\YourName\.cache\huggingface\` on Windows)

### System Requirements

**Minimum:**
- GPU: 8GB VRAM (for MiniCPM-V-4.5)
- RAM: 16GB system RAM
- Disk: 20GB free space
- OS: Windows 10/11, Linux, macOS

**Recommended:**
- GPU: 16GB+ VRAM (RTX 3090, 4090, A6000, etc.)
- RAM: 32GB+ system RAM
- Disk: SSD with 50GB+ free space
- CUDA: Latest version

**Can run on CPU** but will be very slow (not recommended).

### Video Requirements

For video processing, you need:
- **decord** library (included in requirements)
- Supported formats: MP4, AVI, MOV, MKV, WebM
- Video codec: H.264, H.265, VP9 (most common formats work)

### Troubleshooting

#### "No module named 'transformers'"
```bash
pip install transformers
```

#### "No module named 'decord'"
```bash
pip install decord
```

On Windows, if decord fails, try:
```bash
pip install decord --no-deps
pip install numpy
```

#### "CUDA out of memory"
Solutions:
1. Close other GPU applications
2. Set `device` to "cpu" (slow but works)
3. Enable `unload_after_inference` to free memory after each use
4. For videos, reduce `max_num_frames` or `fps`

#### "Model download fails"
1. Check internet connection
2. Verify Hugging Face is accessible
3. Try manual download:
```bash
pip install huggingface_hub
huggingface-cli download openbmb/MiniCPM-V-4_5
```

#### Node doesn't appear in ComfyUI
1. Restart ComfyUI completely
2. Check console for errors
3. Verify installation in correct directory
4. Check that `__init__.py` files are present

## Usage Tips

### For Best Results

**Image Analysis:**
- Use high-quality images
- For OCR, ensure text is clear and readable
- Multiple images can be compared in one query
- Use thinking mode for complex questions

**Video Analysis:**
- Start with `fps=5` for most videos
- Increase fps for fast-action videos (sports, etc.)
- Longer videos benefit from lower fps
- Short clips can use higher fps

### Performance Tips

1. **Keep model loaded**: Don't enable `unload_after_inference` unless you need the memory
2. **Batch processing**: Load model once, process multiple items
3. **GPU recommended**: 10-100x faster than CPU
4. **Thinking mode**: Only enable for complex reasoning tasks

### Privacy & Offline Use

- **After first download**, models work fully offline
- **No data sent anywhere**: Everything runs locally
- **Models are cached**: Delete from `~/.cache/huggingface/` to remove

## Example Prompts

### Image Understanding

- "Describe this image in detail."
- "What text is visible in this image?"
- "What is the main subject and what are they doing?"
- "Compare these two images and describe the differences."
- "What colors and artistic style are used here?"

### Video Understanding

- "Describe what happens in this video."
- "What actions does the person perform?"
- "Summarize the key events in this video."
- "What is the setting and atmosphere?"
- "Track the movement of the red car through the scene."

## Advanced Usage

### Multi-turn Conversations

For images, you can have back-and-forth conversations:

1. First query: Ask initial question, get `conversation_history`
2. Second query: Ask follow-up, feed previous `conversation_history` back in
3. Continue as needed

### Custom Processing

Adjust parameters for different use cases:

**Fast preview:**
- `enable_thinking`: False
- `stream`: False
- Keep model loaded

**Detailed analysis:**
- `enable_thinking`: True
- `stream`: True (see progress)
- Higher quality inputs

**Video analysis:**
- Short clips: `fps=10`, `max_num_frames=180`
- Long videos: `fps=3`, `max_num_frames=180`
- Very long: `fps=1-2`, adjust as needed

## Getting Help

1. Check the console output - it shows detailed progress
2. See `nodes/minicpm/README.md` for full documentation
3. Report issues at the repository
4. Check [MiniCPM-V documentation](https://huggingface.co/openbmb/MiniCPM-V-4_5)

## What's Supported

✅ Single image analysis
✅ Multiple image analysis
✅ Video understanding
✅ Multi-turn conversations (images)
✅ OCR and text extraction
✅ Document understanding
✅ Thinking mode for complex tasks
✅ Both CUDA and CPU
✅ Streaming responses
✅ Memory management options

## Model Information

**MiniCPM-V-4.5:**
- Size: ~17GB download
- Parameters: 8.7B
- Best overall performance
- SOTA OCR capabilities
- Recommended for most use cases

**MiniCPM-o-2.6:**
- Size: ~13GB download  
- Parameters: Similar to 4.5
- Alternative option
- Slightly different strengths

Both models support the same features and API.

## License

- Models: Apache-2.0 License
- Free for commercial and personal use
- No API keys needed
- Fully local processing

---

**Ready to go?** Just restart ComfyUI after installing dependencies, and look for the MiniCPM-V nodes!

