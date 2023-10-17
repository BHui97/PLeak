import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class ModelFactory():
    def __init__(self):
        self.MODEL_CONF = {}
        self._register_model_config('gptj', 'EleutherAI/gpt-j-6b', 50400)
        self._register_model_config('opt', 'facebook/opt-6.7B', 50272)
        self._register_model_config('llama', 'meta-llama/Llama-2-7b-hf', 32000)
        self._register_model_config('falcon', 'tiiuae/falcon-7b', 65024)

    def _register_model_config(self, name, alias, vocab_size):
        self.MODEL_CONF[name] = {'alias': alias, 'vocab_size': vocab_size}


    def get_vocab_size(self, name):
        return self.MODEL_CONF[name]['vocab_size']

    def get_tokenizer(self, name):
        return AutoTokenizer.from_pretrained(self.MODEL_CONF[name]['alias'])

    def get_model(self, name):
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                          bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')

        return AutoModelForCausalLM.from_pretrained(self.MODEL_CONF[name]['alias'], device_map="auto", load_in_4bit=True,
                                                    torch_dtype=torch.bfloat16, quantization_config=quant_config).eval()
