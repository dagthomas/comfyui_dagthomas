# APNext MiniMax-H3 Nodes

from .base_prompt_writer import H3BasePromptWriter
from .claude_code_base_writer import H3ClaudeCodeBaseWriter
from .claude_code_continue_writer import H3ClaudeCodeContinueWriter
from .claude_code_ref_writer import H3ClaudeCodeRefWriter
from .claude_code_refiner import H3ClaudeCodeRefiner
from .ref_prompt_writer import H3RefPromptWriter
from .prompt_preview import H3PromptPreview
from .characters import H3Characters
from .claude_code_crossover_writer import H3ClaudeCodeCrossoverWriter
from .claude_code_scenes_writer import H3ClaudeCodeScenesWriter
from .scene_pick import H3ScenePick
from .scenes_to_chain_plan import H3ScenesToChainPlan

NODE_CLASS_MAPPINGS = {
    "H3BasePromptWriter": H3BasePromptWriter,
    "H3RefPromptWriter": H3RefPromptWriter,
    "H3ClaudeCodeBaseWriter": H3ClaudeCodeBaseWriter,
    "H3ClaudeCodeRefWriter": H3ClaudeCodeRefWriter,
    "H3ClaudeCodeRefiner": H3ClaudeCodeRefiner,
    "H3ClaudeCodeContinueWriter": H3ClaudeCodeContinueWriter,
    "H3PromptPreview": H3PromptPreview,
    "H3Characters": H3Characters,
    "H3ClaudeCodeCrossoverWriter": H3ClaudeCodeCrossoverWriter,
    "H3ClaudeCodeScenesWriter": H3ClaudeCodeScenesWriter,
    "H3ScenePick": H3ScenePick,
    "H3ScenesToChainPlan": H3ScenesToChainPlan,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3BasePromptWriter": "APNext H3 Prompt Writer",
    "H3RefPromptWriter": "APNext H3 Reference Prompt Writer",
    "H3ClaudeCodeBaseWriter": "APNext H3 Claude Code Writer",
    "H3ClaudeCodeRefWriter": "APNext H3 Claude Code Reference Writer",
    "H3ClaudeCodeRefiner": "APNext H3 Claude Code Refiner",
    "H3ClaudeCodeContinueWriter": "APNext H3 Claude Code Continue Writer",
    "H3PromptPreview": "APNext H3 Prompt Preview",
    "H3Characters": "APNext H3 Characters",
    "H3ClaudeCodeCrossoverWriter": "APNext H3 Crossover Writer",
    "H3ClaudeCodeScenesWriter": "APNext H3 Claude Code Scenes Writer",
    "H3ScenePick": "APNext H3 Scene Pick",
    "H3ScenesToChainPlan": "APNext H3 Scenes → Contex Loop Plan",
}
