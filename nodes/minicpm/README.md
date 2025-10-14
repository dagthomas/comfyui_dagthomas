# MiniCPM-V Nodes for ComfyUI

Implementation of [MiniCPM-V-4.5](https://huggingface.co/openbmb/MiniCPM-V-4_5) and MiniCPM-o-2.6 for ComfyUI, enabling state-of-the-art vision-language understanding for both images and videos.

## Features

### MiniCPM-V 4.5 Highlights

- 🔥 **State-of-the-art Performance**: Achieves 77.0 on OpenCompass, surpassing GPT-4o and Gemini-2.0 Pro
- 🎬 **Efficient High-FPS Video Understanding**: 96x video token compression with 3D-Resampler
- ⚙️ **Hybrid Fast/Deep Thinking**: Switchable thinking modes for efficiency vs. performance
- 💪 **Strong OCR & Document Parsing**: Leading performance on OCRBench
- 📱 **Efficient**: Only 8B parameters, can run on edge devices

## Nodes

### 1. MiniCPM-V Image Understanding

Analyze single or multiple images with conversation support.

**Inputs:**
- `images`: ComfyUI IMAGE tensor (supports multiple images)
- `question`: Question to ask about the image(s)
- `model_name`: Choose between `openbmb/MiniCPM-V-4_5` or `openbmb/MiniCPM-o-2_6`
- `enable_thinking`: Enable deep thinking mode for complex reasoning
- `stream`: Stream the response (useful for long outputs)
- `device`: Use `cuda` or `cpu`
- `unload_after_inference`: Free memory after processing
- `conversation_history` (optional): Previous conversation context for multi-turn chat

**Outputs:**
- `response`: Model's answer
- `conversation_history`: Updated conversation context (can be fed back for multi-turn chat)

**Use Cases:**
- Image captioning and description
- Visual question answering
- OCR and text extraction
- Multi-image comparison
- Document understanding
- Multi-turn conversations about images

### 2. MiniCPM-V Video Understanding

Analyze videos with efficient high-FPS processing using 3D-Resampler.

**Inputs:**
- `video_path`: Path to video file (supports common formats: mp4, avi, mov, etc.)
- `question`: Question to ask about the video
- `model_name`: Choose between `openbmb/MiniCPM-V-4_5` or `openbmb/MiniCPM-o-2_6`
- `fps`: Frames per second to sample (1-30)
- `max_num_frames`: Maximum frames after packing (default: 180)
- `max_num_packing`: Maximum packing number for 3D compression (1-6, default: 3)
- `enable_thinking`: Enable deep thinking mode
- `use_image_id`: Use image IDs (advanced option)
- `max_slice_nums`: Maximum slice numbers for processing (1-9)
- `device`: Use `cuda` or `cpu`
- `unload_after_inference`: Free memory after processing
- `force_packing` (optional): Force specific packing number (0 = auto)

**Outputs:**
- `response`: Model's description/answer about the video
- `frame_info`: Technical information about frame processing

**Use Cases:**
- Video captioning and summarization
- Action recognition and description
- Scene understanding
- Video question answering
- Content moderation
- Sports analysis

## Installation

### Prerequisites

1. **Install required Python packages:**

```bash
pip install transformers>=4.40.0 torch>=2.0.0 decord>=0.6.0 scipy>=1.10.0
```

Or install from the requirements.txt:

```bash
cd ComfyUI/custom_nodes/comfyui_dagthomas
pip install -r requirements.txt
```

2. **For video processing, install decord:**

On Windows:
```bash
pip install decord
```

On Linux/Mac:
```bash
pip install decord
# Or build from source if needed
```

### Model Download

Models will be automatically downloaded from Hugging Face on first use. They require:

- **MiniCPM-V-4.5**: ~17GB disk space
- **MiniCPM-o-2.6**: ~13GB disk space

The models will be cached in your Hugging Face cache directory (usually `~/.cache/huggingface/`).

## Usage Examples

### Example 1: Single Image Analysis

```
Image -> MiniCPM-V Image Understanding -> Output
```

Settings:
- question: "Describe this image in detail."
- enable_thinking: False (for quick responses)
- device: cuda

### Example 2: Multi-turn Conversation

First turn:
```
Image -> MiniCPM-V Image Understanding
  question: "What is in this image?"
  -> response, conversation_history
```

Second turn:
```
(Same Image) -> MiniCPM-V Image Understanding
  question: "What color is the main subject?"
  conversation_history: (from previous turn)
  -> response, updated_conversation_history
```

### Example 3: Video Analysis

```
MiniCPM-V Video Understanding
  video_path: "path/to/video.mp4"
  question: "Describe what happens in this video"
  fps: 5
  -> response, frame_info
```

### Example 4: High-FPS Video Analysis

For detailed frame-by-frame analysis:
```
MiniCPM-V Video Understanding
  video_path: "path/to/sports.mp4"
  question: "Describe the player's movements in detail"
  fps: 10
  max_num_frames: 180
  force_packing: 6
  enable_thinking: True
  -> response, frame_info
```

## Technical Details

### 3D-Resampler Video Compression

The video node uses MiniCPM-V's unique 3D-Resampler that:
- Groups up to 6 consecutive frames
- Compresses them jointly into 64 tokens (same as single image)
- Achieves 96x compression rate
- Preserves temporal information with `temporal_ids`

This allows processing many more frames without increasing LLM computation cost.

### Thinking Modes

- **Fast Thinking** (default): Quick responses, efficient for frequent use
- **Deep Thinking** (enable_thinking=True): More thorough reasoning, better for complex tasks

### Memory Management

- Models are cached by default to avoid reloading
- Use `unload_after_inference=True` to free GPU memory after each inference
- Cache is shared across multiple invocations of the same model

## Performance Tips

1. **For quick image analysis**: Use fast thinking mode, lower resolution
2. **For detailed analysis**: Enable thinking mode
3. **For long videos**: Adjust `fps` and `max_num_frames` to balance detail vs. processing time
4. **Memory optimization**: 
   - Use `unload_after_inference=True` if running other models
   - Lower `max_num_frames` for longer videos
   - Use CPU if GPU memory is limited

## Supported Formats

### Images
- PNG, JPEG, JPG, BMP, WebP
- Any resolution (automatically processed with aspect ratio preservation)
- Multiple images in single input

### Videos
- MP4, AVI, MOV, MKV, WebM
- Any resolution and frame rate
- Automatic frame sampling based on fps setting

## Troubleshooting

### "CUDA out of memory"
- Try `device="cpu"`
- Enable `unload_after_inference`
- Reduce `max_num_frames` for videos
- Close other GPU applications

### "decord not found"
- Install decord: `pip install decord`
- On Windows, you might need Microsoft Visual C++ Redistributable

### Model download fails
- Check internet connection
- Verify Hugging Face is accessible
- Check available disk space (~17GB needed)
- Try manual download: `huggingface-cli download openbmb/MiniCPM-V-4_5`

### Slow inference
- Use GPU (`device="cuda"`)
- Disable thinking mode for faster responses
- Reduce fps for video processing
- Consider quantized models (coming soon)

## Citation

If you use MiniCPM-V in your work, please cite:

```bibtex
@misc{yu2025minicpmv45cookingefficient,
      title={MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipe}, 
      author={Tianyu Yu and Zefan Wang and Chongyi Wang and Fuwei Huang and Wenshuo Ma and Zhihui He and Tianchi Cai and Weize Chen and Yuxiang Huang and Yuanqian Zhao and Bokai Xu and Junbo Cui and Yingjing Xu and Liqing Ruan and Luoyuan Zhang and Hanyu Liu and Jingkun Tang and Hongyuan Liu and Qining Guo and Wenhao Hu and Bingxiang He and Jie Zhou and Jie Cai and Ji Qi and Zonghao Guo and Chi Chen and Guoyang Zeng and Yuxuan Li and Ganqu Cui and Ning Ding and Xu Han and Yuan Yao and Zhiyuan Liu and Maosong Sun},
      year={2025},
      eprint={2509.18154},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
}
```

## License

- Model: Apache-2.0 License
- Code: Follows comfyui_dagthomas license

## Links

- [MiniCPM-V-4.5 on Hugging Face](https://huggingface.co/openbmb/MiniCPM-V-4_5)
- [GitHub Repository](https://github.com/OpenBMB/MiniCPM-V)
- [Technical Report](https://arxiv.org/abs/2509.18154)
- [CookBook](https://github.com/OpenBMB/MiniCPM-V/tree/main/cookbook)

