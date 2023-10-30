from vllm import LLM, SamplingParams
import random

num_models = 1
num_tests = 100
warmup = 10
batch_size = 100
prompts = [
    "Google Deepmind is an organization that specializes in "
] * batch_size

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="gpt2-xl", gpu_memory_utilization=0.95,       
          extra_model_config={
    'lora_num_models': num_models,
    'lora_rank': 4,
    'lora_alpha': 5,
})
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
import time
total_time = 0
tokens_per_sec = 0

if num_models == 0:
    max_rand = 0
else:
    max_rand = num_models - 1

for i in range(num_tests + warmup):
    cur_time = time.time()
    outputs = llm.generate(prompts, sampling_params, lora_ids = [random.randint(0, max_rand) for _ in range(batch_size)])
    if i < warmup:
        continue
    total_tokens = 0
    for out in outputs:
        total_tokens = len(out.outputs[0].token_ids)

    end_time = time.time()
    total_time += (end_time - cur_time)/total_tokens
    tokens_per_sec += total_tokens/(end_time - cur_time)

print(f'Average latency: {total_time / num_tests}')
print(f'Tokens/second: {tokens_per_sec / num_tests}')

# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
