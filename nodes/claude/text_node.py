# Claude Text Node

import os
import random

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

from ...utils.constants import CUSTOM_CATEGORY, claude_models


class ClaudeTextNode:
    def __init__(self):
        self.client = None  # Lazy initialization

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_text": ("STRING", {"multiline": True}),
                "happy_talk": ("BOOLEAN", {"default": True}),
                "compress": ("BOOLEAN", {"default": False}),
                "compression_level": (["soft", "medium", "hard"],),
                "poster": ("BOOLEAN", {"default": False}),
                "claude_model": (claude_models, {"default": "claude-3-5-sonnet-20241022"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "randomize_each_run": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_base_prompt": ("STRING", {"multiline": True, "default": ""}),
                "custom_title": ("STRING", {"default": ""}),
                "override": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "variation_instruction": (
                    "STRING", 
                    {"multiline": True, "default": "Generate different creative variations each time while maintaining the core concept."}
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = f"{CUSTOM_CATEGORY}/LLM"

    def generate(
        self,
        input_text,
        happy_talk,
        compress,
        compression_level,
        poster,
        claude_model="claude-3-5-sonnet-20241022",
        seed=-1,
        randomize_each_run=True,
        custom_base_prompt="",
        custom_title="",
        override="",
        variation_instruction="Generate different creative variations each time while maintaining the core concept.",
    ):
        # Set random seed for consistent randomization within this call
        if seed != -1:
            current_seed = seed
        elif randomize_each_run:
            current_seed = random.randint(0, 2**32 - 1)
        else:
            current_seed = 42  # Fixed seed for consistent results
            
        # Set random seed for consistent randomization within this call
        random.seed(current_seed)
        print(f"🧠 Claude using seed: {current_seed}")
        
        # Build the prompt
        if override:
            prompt = override
        else:
            if custom_base_prompt:
                base_prompt = custom_base_prompt
            else:
                base_prompt = self._get_base_prompt(happy_talk, compress, compression_level, poster)
            
            if custom_title:
                prompt = f"{base_prompt}\n\nTitle: {custom_title}\n\nContent: {input_text}"
            else:
                prompt = f"{base_prompt}\n\nContent: {input_text}"
            
            # Add variation instruction for randomness
            if randomize_each_run and variation_instruction:
                prompt += f"\n\nVariation Instruction: {variation_instruction}"

        try:
            # Lazy initialization of client
            if self.client is None:
                if not ANTHROPIC_AVAILABLE:
                    return ("Error: anthropic package not installed. Install with: pip install anthropic",)
                
                api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
                if not api_key:
                    return ("Error: ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable is not set",)
                
                self.client = anthropic.Anthropic(api_key=api_key)
            
            response = self.client.messages.create(
                model=claude_model,
                max_tokens=2000,
                temperature=0.7 if randomize_each_run else 0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = response.content[0].text.strip()
            print(f"🧠 Claude ({claude_model}) generated {len(result)} characters")
            
            return (result,)
            
        except Exception as e:
            error_msg = f"Claude API Error: {str(e)}"
            print(f"❌ {error_msg}")
            return (error_msg,)

    def _get_base_prompt(self, happy_talk, compress, compression_level, poster):
        """Generate base prompt based on settings"""
        base_prompt = "You are Claude, an AI assistant created by Anthropic. You are an expert creative writing assistant. "
        
        if happy_talk:
            base_prompt += "Write in an enthusiastic, positive, and engaging tone. "
        else:
            base_prompt += "Write in a clear, direct, and professional tone. "
        
        if compress:
            if compression_level == "soft":
                base_prompt += "Keep the response concise while maintaining key details. "
            elif compression_level == "medium":
                base_prompt += "Provide a moderately compressed response focusing on essential information. "
            elif compression_level == "hard":
                base_prompt += "Give a very brief and highly compressed response with only the most critical points. "
        
        if poster:
            base_prompt += "Format the output as if it were promotional or poster-style content with impactful language. "
        
        base_prompt += "Enhance and expand the following content:"
        
        return base_prompt
