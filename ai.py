import os
import json
import threading
import requests
from llama_cpp import Llama

# ollama config
OLLAMA_URL = "http://176.146.128.50:11434"
REQUIRED_MODELS = ["ministral-3:3b"]

# init model
qwen_model = Llama(
    model_path="Qwen3.5-0.8B-Q4_K_M.gguf", 
    n_ctx=2048, 
    verbose=False
)

def _pull_model_async(model_name):
    # pull model
    try:
        requests.post(f"{OLLAMA_URL}/api/pull", json={"name": model_name})
    except:
        pass

def _preload_model_async(model_name):
    # load model
    try:
        requests.post(f"{OLLAMA_URL}/api/generate", json={"model": model_name, "keep_alive": "10m"})
    except:
        pass

def verify_and_pull_models():
    # verify models
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            data = response.json()
            installed = [m["name"] for m in data.get("models", [])]
            
            for req_model in REQUIRED_MODELS:
                if req_model not in installed:
                    # pull async
                    thread = threading.Thread(target=_pull_model_async, args=(req_model,))
                    thread.start()
                else:
                    # preload async
                    thread = threading.Thread(target=_preload_model_async, args=(req_model,))
                    thread.start()
    except requests.exceptions.ConnectionError:
        # handle error
        print("ollama offline")

def analyze_intent(messages):
    # analyze intent
    system_prompt = (
        "classifie le message de l'utilisateur strictement dans l'une de ces catégories : "
        "discussion, discussion_long, simple_tool, complex_tool, invent_tool. "
        "retourne uniquement le nom exact de la catégorie sans aucun autre texte."
    )
    
    # format prompt
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for msg in messages:
        prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    
    try:
        # call local
        response = qwen_model(prompt, max_tokens=10, stop=["<|im_end|>"])
        intent = response["choices"][0]["text"].strip().lower()
        
        valid_intents = ["discussion", "discussion_long", "simple_tool", "complex_tool", "invent_tool"]
        if intent in valid_intents:
            return intent
    except:
        pass
        
    return "discussion"

def route_llm_request(messages, tools=None):
    # analyze intent
    intent = analyze_intent(messages)
    
    # print intent
    print(f"intent: {intent}")
    
    # map config
    routes = {
        "discussion": {
            "model": "ministral-3:3b", 
            "provider": "ollama",
            "prompt": "réponds obligatoirement et uniquement en français de manière très courte et naturelle."
        },
        "simple_tool": {
            "model": "local", 
            "provider": "local",
            "prompt": "tu es un assistant technique. analyse la demande pour utiliser un outil simple de manière concise."
        },
        "discussion_long": {
            "model": "mistral-medium", 
            "provider": "mistral",
            "prompt": "tu es un expert. fournis une explication détaillée en français, claire pour une écoute vocale."
        },
        "invent_tool": {
            "model": "devstral-small-2", 
            "provider": "mistral",
            "prompt": "tu es un développeur expert. conçois du code fonctionnel."
        },
        "complex_tool": {
            "model": "mistral-large-latest", 
            "provider": "mistral",
            "prompt": "tu es un orchestrateur avancé. résous les problèmes complexes."
        }
    }
    
    target = routes.get(intent, routes["discussion"])
    model = target["model"]
    provider = target["provider"]
    model_prompt = target["prompt"]
    
    # merge system prompts
    routed_messages = []
    combined_system = model_prompt
    
    for msg in messages:
        if msg["role"] == "system":
            combined_system = msg["content"] + " " + combined_system
        else:
            routed_messages.append(msg)
            
    routed_messages.insert(0, {"role": "system", "content": combined_system})
    
    def generate_stream():
        if provider == "local":
            # format prompt
            prompt = ""
            for msg in routed_messages:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            
            # call local
            stream = qwen_model(prompt, stream=True, max_tokens=512, stop=["<|im_end|>"])
            for chunk in stream:
                text = chunk["choices"][0].get("text", "")
                if text:
                    yield text
                    
        elif provider == "ollama":
            # call ollama
            payload = {
                "model": model,
                "messages": routed_messages,
                "stream": True,
                "keep_alive": "10m" 
            }
            if tools:
                payload["tools"] = tools
                
            res = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True)
            
            # parse ollama
            for line in res.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
        
        elif provider == "mistral":
            # call mistral
            api_key = os.getenv("MISTRAL_API_KEY", "")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": routed_messages,
                "stream": True
            }
            if tools:
                payload["tools"] = tools
                
            res = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, stream=True)
            
            # parse mistral
            for line in res.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: ") and line_str != "data: [DONE]":
                        data = json.loads(line_str[6:])
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                                
    # return model and stream
    return model, generate_stream()