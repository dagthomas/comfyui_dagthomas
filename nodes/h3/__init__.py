# APNext MiniMax-H3 Nodes

from .base_prompt_writer import H3BasePromptWriter
from .claude_code_base_writer import H3ClaudeCodeBaseWriter
from .claude_code_continue_writer import H3ClaudeCodeContinueWriter
from .claude_code_ref_writer import H3ClaudeCodeRefWriter
from .claude_code_refiner import H3ClaudeCodeRefiner
from .ref_prompt_writer import H3RefPromptWriter
from .prompt_preview import H3PromptPreview

NODE_CLASS_MAPPINGS = {
    "H3BasePromptWriter": H3BasePromptWriter,
    "H3RefPromptWriter": H3RefPromptWriter,
    "H3ClaudeCodeBaseWriter": H3ClaudeCodeBaseWriter,
    "H3ClaudeCodeRefWriter": H3ClaudeCodeRefWriter,
    "H3ClaudeCodeRefiner": H3ClaudeCodeRefiner,
    "H3ClaudeCodeContinueWriter": H3ClaudeCodeContinueWriter,
    "H3PromptPreview": H3PromptPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3BasePromptWriter": "APNext H3 Prompt Writer",
    "H3RefPromptWriter": "APNext H3 Reference Prompt Writer",
    "H3ClaudeCodeBaseWriter": "APNext H3 Claude Code Writer",
    "H3ClaudeCodeRefWriter": "APNext H3 Claude Code Reference Writer",
    "H3ClaudeCodeRefiner": "APNext H3 Claude Code Refiner",
    "H3ClaudeCodeContinueWriter": "APNext H3 Claude Code Continue Writer",
    "H3PromptPreview": "APNext H3 Prompt Preview",
}
