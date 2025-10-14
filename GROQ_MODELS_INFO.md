# Groq Models Information

## 🚀 Dynamic Model Loading

The Groq integration now supports **dynamic model loading** from the Groq API! When you have a valid `GROQ_API_KEY` set, the system will automatically fetch the latest available models on startup.

### How It Works

1. **At Startup**: The system tries to fetch models from `https://api.groq.com/openai/v1/models`
2. **Fallback**: If the API call fails (no key, network issue), it loads from `data/groq_models.json`
3. **Cache**: Models are loaded once per ComfyUI session

### Benefits

- ✅ Always have access to the latest models
- ✅ Automatically get new models as Groq adds them
- ✅ No manual updates needed
- ✅ Graceful fallback if API is unavailable

---

## 📋 Currently Available Models (as of API check)

### 🔤 Text/Chat Models

| Model ID | Provider | Context Window | Max Tokens | Description |
|----------|----------|----------------|------------|-------------|
| `llama-3.3-70b-versatile` | Meta | 131K | 32K | Latest Llama 3.3, very capable |
| `llama-3.1-8b-instant` | Meta | 131K | 131K | Fast, efficient Llama 3.1 |
| `meta-llama/llama-4-scout-17b-16e-instruct` | Meta | 131K | 8K | ⭐ New Llama 4 Scout model |
| `meta-llama/llama-4-maverick-17b-128e-instruct` | Meta | 131K | 8K | ⭐ New Llama 4 Maverick model |
| `groq/compound` | Groq | 131K | 8K | 🔥 Groq's proprietary model |
| `groq/compound-mini` | Groq | 131K | 8K | Groq's efficient model |
| `openai/gpt-oss-120b` | OpenAI | 131K | 65K | OpenAI's open source 120B model |
| `openai/gpt-oss-20b` | OpenAI | 131K | 65K | OpenAI's open source 20B model |
| `moonshotai/kimi-k2-instruct` | Moonshot AI | 131K | 16K | Kimi K2 model |
| `moonshotai/kimi-k2-instruct-0905` | Moonshot AI | 262K | 16K | 🚀 Kimi with 262K context! |
| `qwen/qwen3-32b` | Alibaba Cloud | 131K | 40K | Qwen 3 model |
| `allam-2-7b` | SDAIA | 4K | 4K | ALLAM model |

### 👁️ Vision Models

**Note**: Vision model availability may vary by region and account tier. The API response showed no vision models in the current check, but Groq has supported:
- `llama-3.2-90b-vision-preview`
- `llama-3.2-11b-vision-preview`

If vision models aren't available for your account, the vision node will gracefully handle this.

### 🎙️ Other Models (Not included in text generation)

The API also returns:
- **Whisper** models for speech-to-text
- **PlayAI TTS** models for text-to-speech
- **Prompt Guard** models for safety filtering

These are filtered out from the text generation node as they serve different purposes.

---

## 🔄 Updating Models Manually

If you want to manually update the models list, you can:

### Method 1: Use the Update Script

```bash
# Set your API key
export GROQ_API_KEY=your_key_here

# Run the updater
python utils/update_groq_models.py
```

### Method 2: Use cURL + Python

```bash
# Fetch models
curl -X GET "https://api.groq.com/openai/v1/models" \
     -H "Authorization: Bearer $GROQ_API_KEY" \
     -H "Content-Type: application/json" > groq_models_raw.json

# Parse and update (you'll need to manually edit the JSON)
```

### Method 3: Restart ComfyUI

Simply restart ComfyUI with your `GROQ_API_KEY` set, and the system will automatically fetch the latest models!

---

## 🎯 Recommended Models for Different Tasks

### For Creative Writing
- `llama-3.3-70b-versatile` - Best overall quality
- `groq/compound` - Groq's optimized model
- `meta-llama/llama-4-maverick-17b-128e-instruct` - New Llama 4

### For Fast Generation
- `llama-3.1-8b-instant` - Very fast with good quality
- `groq/compound-mini` - Groq's fast model

### For Long Context
- `moonshotai/kimi-k2-instruct-0905` - 262K context window!
- `llama-3.3-70b-versatile` - 131K context

### For Code Generation
- `openai/gpt-oss-120b` - Strong coding capabilities
- `qwen/qwen3-32b` - Good for code

---

## 💡 Tips

1. **API Key**: Get your free API key at https://console.groq.com/
2. **Speed**: Groq is known for extremely fast inference (up to 750 tokens/sec)
3. **Cost**: Very competitive pricing, especially for open-source models
4. **Context**: Many models support 131K+ tokens of context
5. **Updates**: Models list may change; system will auto-update on restart

---

## 🔧 Configuration

### Environment Variable
```bash
export GROQ_API_KEY=gsk_your_key_here
```

### In ComfyUI
1. Set the environment variable before starting ComfyUI
2. Restart ComfyUI to load the latest models
3. Check the console for model loading messages:
   - ✅ "Fetched X Groq models from API" = Success!
   - 📋 "Loaded X Groq models from JSON file" = Fallback mode

---

## 🆚 Model Comparison with Other Providers

| Feature | Groq Models | GPT-4 | Claude | Gemini |
|---------|-------------|-------|--------|--------|
| Speed | ⚡⚡⚡ Very Fast | Medium | Medium | Fast |
| Context | Up to 262K | 128K | 200K | 2M |
| Cost | $ Very Low | $$$ High | $$ Medium | $ Low |
| Open Source | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Vision | ⚠️ Limited | ✅ Yes | ✅ Yes | ✅ Yes |
| Quality | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 🐛 Troubleshooting

### Models not loading?
- Check your API key is valid
- Check internet connection
- Look for error messages in ComfyUI console
- Fallback JSON file should still work

### Vision models not showing?
- Vision model availability varies by region
- Check Groq's documentation for current vision model support
- Try using text models with detailed descriptions as alternative

### Need to force refresh models?
- Restart ComfyUI with `GROQ_API_KEY` set
- Or run `utils/update_groq_models.py`

---

## 📚 Resources

- **Groq Console**: https://console.groq.com/
- **Groq Documentation**: https://console.groq.com/docs/
- **API Reference**: https://console.groq.com/docs/api-reference
- **Pricing**: https://groq.com/pricing/

