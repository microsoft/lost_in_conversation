from openai import OpenAI, AzureOpenAI
import os, time, json, re


def validate_model_lock(request_model, served_model=None):
    """Enforce an optional request/response model allow-list."""
    allowed_model = os.environ.get("LOST_CONVERSATION_ALLOWED_MODEL", "").strip()
    if not allowed_model:
        return
    if request_model != allowed_model:
        raise ValueError(
            f"Request model {request_model!r} does not match locked model "
            f"{allowed_model!r}"
        )
    if served_model is not None and served_model != allowed_model:
        raise RuntimeError(
            f"Served model {served_model!r} does not match locked model "
            f"{allowed_model!r}"
        )

def format_messages(messages, variables={}):
    last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]

    for k, v in variables.items():
        key_string = f"[[{k}]]"
        if key_string not in last_user_msg["content"]:
            print(f"[prompt] Key {k} not found in prompt; effectively ignored")
        assert type(v) == str, f"[prompt] Variable {k} is not a string"
        last_user_msg["content"] = last_user_msg["content"].replace(key_string, v)

    # find all the keys that are still in the prompt using regex [[STR]] where STR is alnum witout space
    keys_still_in_prompt = re.findall(r"\[\[([^\]]+)\]\]", last_user_msg["content"])
    if len(keys_still_in_prompt) > 0:
        print(f"[prompt] The following keys were not replaced: {keys_still_in_prompt}")

    return messages

class OpenAI_Model:
    def __init__(self):
        if "AZURE_OPENAI_API_KEY" in os.environ and "AZURE_OPENAI_ENDPOINT" in os.environ:
            self.client = AzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"], azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_version="2024-10-01-preview")
        else:
            assert "OPENAI_API_KEY" in os.environ
            self.client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )

    def cost_calculator(self, model, usage, is_batch_model=False):
        is_finetuned, base_model = False, model
        if model.startswith("ft:gpt"):
            is_finetuned = True
            base_model = model.split(":")[1]

        prompt_tokens = usage['prompt_tokens']
        if 'prompt_tokens_details' in usage:
            prompt_tokens_cached = usage['prompt_tokens_details']['cached_tokens']
        else:
            prompt_tokens_cached = 0
        prompt_tokens_non_cached = prompt_tokens - prompt_tokens_cached

        completion_tokens = usage['completion_tokens']
        if base_model.startswith("gpt-4o-mini"):
            if is_finetuned:
                inp_token_cost, out_token_cost = 0.0003, 0.00015
            else:
                inp_token_cost, out_token_cost = 0.00015, 0.0006
        elif base_model.startswith("gpt-4o"):
            if is_finetuned:
                inp_token_cost, out_token_cost = 0.00375, 0.015
            else:
                inp_token_cost, out_token_cost = 0.0025, 0.01
        elif base_model.startswith("gpt-3.5-turbo"):
            inp_token_cost, out_token_cost = 0.0005, 0.0015
        elif base_model.startswith("o1-mini"):
            inp_token_cost, out_token_cost = 0.003, 0.012
        elif base_model.startswith("gpt-4.5-preview"):
            inp_token_cost, out_token_cost = 0.075, 0.150
        elif base_model.startswith("o1-preview") or base_model == "o1":
            inp_token_cost, out_token_cost = 0.015, 0.06
        else:
            return 0.0

        cache_discount = 0.5 # cached tokens are half the price
        batch_discount = 0.5 # batch API is half the price
        total_usd = ((prompt_tokens_non_cached + prompt_tokens_cached * cache_discount) / 1000) * inp_token_cost + (completion_tokens / 1000) * out_token_cost
        if is_batch_model:
            total_usd *= batch_discount

        return total_usd

    def generate(self, messages, model="gpt-4o-mini", timeout=30, max_retries=3, temperature=1.0, is_json=False, return_metadata=False, max_tokens=None, variables={}):
        validate_model_lock(model)
        kwargs = {}
        if is_json:
            kwargs["response_format"] = { "type": "json_object" }
        N = 0

        messages = format_messages(messages, variables)

        # o1- models do not support system message. If the first message is a system message, and the second message is a user message, then prepend the user message with the system message.
        if model.startswith("o1") and len(messages) > 1 and messages[0]["role"] == "system" and messages[1]["role"] == "user":
            system_message = messages[0]["content"]
            messages[1]["content"] = f"System Message: {system_message}\n{messages[1]['content']}"
            messages = messages[1:]

        while True:
            try:
                response = self.client.chat.completions.create(model=model, messages=messages, timeout=timeout, max_completion_tokens=max_tokens, temperature=temperature, **kwargs)
                break
            except:
                N += 1
                if N >= max_retries:
                    raise Exception("Failed to get response from OpenAI")
                else:
                    time.sleep(4)

        response = response.to_dict()
        validate_model_lock(model, response.get("model"))
        usage = response['usage']
        response_text = response["choices"][0]["message"]["content"]
        total_usd = self.cost_calculator(model, usage)
        prompt_tokens_cached = 0
        if 'prompt_tokens_details' in usage:
            prompt_tokens_cached = usage['prompt_tokens_details']['cached_tokens']

        if not return_metadata:
            return response_text
        return {"message": response_text, "request_model": model, "served_model": response.get("model"), "total_tokens": usage['total_tokens'], "prompt_tokens": usage['prompt_tokens'], "prompt_tokens_cached": prompt_tokens_cached, "completion_tokens": usage['completion_tokens'], "total_usd": total_usd}

    def generate_json(self, messages, model="gpt-4o-mini", **kwargs):
        response = self.generate(messages, model, is_json=True, **kwargs)
        response["message"] = json.loads(response["message"])
        return response


model = OpenAI_Model()
generate = model.generate
generate_json = model.generate_json





if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Humor is a way to make people laugh. Tell me a joke about UC Berkeley."},
        {"role": "assistant", "content": '{"joke": '}
    ]

    model = "gpt-4o"
    response = generate(messages, model=model, return_metadata=True)
    
    print(response)

    # batch_id = client.generate_w_delay(messages, model="gpt-4o-mini")
    # print(batch_id)
