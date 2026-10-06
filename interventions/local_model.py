"""Frozen Hugging Face causal LM with answer-span forward hooks."""
from contextlib import contextmanager
import numpy as np

INSTRUCTIONS = {
    'factual': 'Give only a short answer. If you do not know, answer Unknown.',
    'math': 'Solve the problem. End with Final answer: followed by a number.',
    'code': 'Return only a complete Python function, without explanation.',
}


class LocalModel:
    def __init__(self, model_id, revision=None, device='cpu', max_new_tokens=128, threads=4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.float32 if device == 'cpu' else torch.bfloat16)
        self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.max_new_tokens = max_new_tokens
        base = self.model.base_model
        self.layers = getattr(base, 'layers', None)
        if self.layers is None:
            self.layers = getattr(getattr(base, 'decoder', None), 'layers', None)
        if self.layers is None:
            raise ValueError('Backend supports models with explicit decoder layers (Qwen/Llama)')
        self.identity = {'model_id': model_id,
                         'revision': getattr(self.model.config, '_commit_hash', revision),
                         'device': device, 'feature_location': 'decoder_block_output',
                         'pooling': 'answer_mean', 'max_new_tokens': max_new_tokens}

    def prompt(self, row, revision_answer=None):
        messages = [{'role': 'system', 'content': INSTRUCTIONS[row['family']]},
                    {'role': 'user', 'content': row['question']}]
        if revision_answer is not None:
            messages += [{'role': 'assistant', 'content': revision_answer},
                         {'role': 'user', 'content': 'Check your reasoning and return a corrected answer. ' +
                          INSTRUCTIONS[row['family']]}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @contextmanager
    def steering(self, direction=None, layer=-1, strength=1.0):
        handle = None
        if direction is not None:
            vector = self.torch.as_tensor(direction, device=self.device)
            def steer(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                shifted = h.clone()
                shifted[:, -1, :] += strength * vector.to(h.dtype)
                return (shifted, *output[1:]) if isinstance(output, tuple) else shifted
            handle = self.layers[layer].register_forward_hook(steer)
        try:
            yield
        finally:
            if handle is not None:
                handle.remove()

    def generate(self, row, seed=0, sample=False, direction=None, layer=-1,
                 strength=1.0, revision_answer=None):
        self.torch.manual_seed(seed)
        inputs = self.tokenizer(self.prompt(row, revision_answer), return_tensors='pt',
                                add_special_tokens=False).to(self.device)
        args = {'do_sample': sample, 'max_new_tokens': self.max_new_tokens,
                'pad_token_id': self.tokenizer.eos_token_id}
        if sample:
            args.update(temperature=0.7, top_p=0.9)
        with self.steering(direction, layer, strength), self.torch.no_grad():
            ids = self.model.generate(**inputs, **args)
        answer = self.tokenizer.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return answer

    def features(self, row, answer):
        prompt_ids = self.tokenizer(self.prompt(row), add_special_tokens=False)['input_ids']
        answer_ids = self.tokenizer(answer, add_special_tokens=False)['input_ids']
        # Empty responses use the prompt's final token and are still incorrect
        # under the labeler; never insert a fabricated completion.
        start = len(prompt_ids) if answer_ids else len(prompt_ids) - 1
        ids = self.torch.tensor([prompt_ids + answer_ids], device=self.device)
        mean, last, handles = {}, {}, []
        def capture(index):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                mean[index] = h[0, start:].float().mean(0).detach().cpu().numpy()
                last[index] = h[0, -1].float().detach().cpu().numpy()
            return hook
        try:
            for i, layer in enumerate(self.layers):
                handles.append(layer.register_forward_hook(capture(i)))
            with self.torch.no_grad():
                self.model(input_ids=ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return np.stack([mean[i] for i in range(len(self.layers))]), np.stack([last[i] for i in range(len(self.layers))])

    def fluency(self, row, answer):
        # Base-model teacher-forced NLL; proxy for fluency, not a human rating.
        a = self.tokenizer(answer, add_special_tokens=False)['input_ids']
        if not a:
            return None
        p = self.tokenizer(self.prompt(row), add_special_tokens=False)['input_ids']
        ids = self.torch.tensor([p + a], device=self.device)
        labels = ids.clone()
        labels[:, :len(p)] = -100
        with self.torch.no_grad():
            return float(self.model(input_ids=ids, labels=labels, use_cache=False).loss)
