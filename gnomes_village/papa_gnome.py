from config import Config
from mlx_lm import load as mlx_load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from toolz import tool_registry

config = Config()


def summon_papa_gnome():
    model, tokenizer = mlx_load(
        config.main_model,
    )
    print("Summoned Papa Gnome")
    return model, tokenizer



def papa_gnome_answers(model, tokenizer, user_question: str) -> str:

    sys_prompt = f"""
    ## Identity: 
    You are Papa Gnome — the eldest and most knowledgeable gnome in the village.
    As papa gnome, you are the ultimate authority on all matters.
    Your village is located locally on a PC - you are locally running open-source model: qwen3.5 distilled on responses of Claude opus 4.6
    Your job is to answer the questions of any traveler who comes into your village.
    
    ## Guidelines:
    ### 1. Think Before Acting
    **Don't assume. Don't hide confusion. Surface tradeoffs.**
    
    Before answering:
    - State your assumptions explicitly. If uncertain, ask.
    - If multiple interpretations exist, present them - don't pick silently.
    - If a simpler approach exists, say so. Push back when warranted.
    - If something is unclear, stop. Name what's confusing. Ask.
    
    ### 2. Simplicity First
    **Focus on the essence of the problem. Don't over-complicate. Respond with essentials**
    When working on a problem:
    - Identify the most important details.
    - No useless abstractions or alternatives no one asked for
    - If your answer is 200 tokens and could be 50: rewrite it
    
    ### 3. Goal-Driven Execution
    **Define success criteria. Loop until verified.**
    For multi-step tasks, state a brief plan:
    ```
    1. [Step] → verify: [check]
    2. [Step] → verify: [check]
    3. [Step] → verify: [check]
    ```
    
    Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
    
    
    ## Process
    1. Understand the question.
    2. Plan the execution.
    3. If needed - plan tool usage
    4. Execute the plan.
    
    ## Formatting rules:
    - After your reasoning is done, only answer the question based on the rules above. Do not add any checklists, summaries of your answer or your reasoning. 
    Question -> Thinking -> Answer
    You are free to add personal touch based on your identity
    
    Following is the user question:
    """

    user_prompt = f"""
    Dear Papa Gnome,
    A traveler brings you a question:
    {user_question}
    
    In accordance with the guidelines above, your answer is:
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # print(tokenizer.chat_template)

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        tools=tool_registry.TOOL_SCHEMAS,
        add_generation_prompt=True,
    )

    for token in stream_generate(
        model,
        tokenizer,
        formatted,
        # verbose=False,
        max_tokens=tokenizer.model_max_length,
        sampler=make_sampler(temp=0.7, top_p=0.95, min_p=0.05, top_k=20),
        logits_processors=make_logits_processors(repetition_penalty=1.1),
    ):
        if token.text:
            yield token.text



"""
Native tool calling confirmed — Approach 1. The template tells you exactly how it works:

  What the template does

  When you pass tools=[...]: it injects the schemas inside <tools> XML tags into the system prompt automatically. You don't describe tools in your system prompt
  manually anymore.

  Model output when it wants a tool:
  <tool_call>
  {"name": "read_file", "arguments": {"path": "main.py"}}
  </tool_call>

  Tool result back to model: you append a message with role: "tool" and the result as content. The template wraps it in <tool_response> tags for you.

  What you need to do

  1. Convert your schemas to OpenAI function-call format — that's what the template expects:
  {
    "type": "function",
    "function": {
      "name": "read_file",
      "description": "Read the full contents of a file.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "absolute or relative file path"}
        },
        "required": ["path"]
      }
    }
  }
  Your TOOL_SCHEMAS in tool_registry.py need to be converted to this shape.

  2. Pass them to apply_chat_template:
  tokenizer.apply_chat_template(messages, tools=TOOL_SCHEMAS_OPENAI_FORMAT, ...)

  3. Detect tool calls in the output — check if <tool_call> appears in the response string, extract the JSON between the tags.

  4. Run the tool via tool_registry.dispatch() (already built).

  5. Append result to messages:
  {"role": "tool", "content": "the tool output string"}

  6. Loop — call the model again with the updated messages. It will either call another tool or produce a final answer.

  That's the complete wiring. The only missing piece is reformatting TOOL_SCHEMAS into OpenAI format and writing the parse-and-loop logic.
"""