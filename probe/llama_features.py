from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class QwenResidualFeatureExtractor:
    """Extracts mean-pooled residual-stream (hidden state) activations from
    a Qwen2.5 decoder layer, for use directly as classifier input. No SAE —
    the raw d_model-dim vector is the feature vector."""

    def __init__(
        self,
        model_id="Qwen/Qwen2.5-7B-Instruct",
        layer=20,
        device="cuda",
    ):
        # Hard requirement: refuse to run on CPU. "auto" device_map (the
        # previous default) can silently offload layers to CPU/disk when
        # GPU memory is tight, which we don't want here.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "QwenResidualFeatureExtractor requires a CUDA GPU, but "
                "torch.cuda.is_available() is False. Refusing to fall back "
                "to CPU."
            )
        if device != "cuda" and not str(device).startswith("cuda"):
            raise ValueError(
                f"device={device!r} is not a CUDA device. This class only "
                "supports running on GPU."
            )

        self.layer = layer
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # left padding is required so that, after generation, every sequence
        # in a batch has its prompt ending at the same index (needed to
        # slice out the generated-answer span uniformly across the batch)
        self.tokenizer.padding_side = "left"

        # Pin everything to the requested GPU explicitly instead of
        # device_map="auto", which is allowed to spill layers onto CPU/disk
        # under memory pressure. Use an explicit index (not bare "cuda") and
        # cap max_memory on that device so bitsandbytes/accelerate can never
        # silently reassign any module to cpu/disk — it will raise an OOM
        # instead if the model doesn't fit.
        cuda_index = torch.cuda.current_device() if device == "cuda" else int(str(device).split(":")[-1])
        gpu_key = f"cuda:{cuda_index}"
        total_mem_gb = torch.cuda.get_device_properties(cuda_index).total_memory / (1024**3)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": cuda_index},
            max_memory={cuda_index: f"{total_mem_gb:.1f}GiB"},
        )
        self.model.eval()

        # sanity-check the requested layer exists on this model before any
        # hook is registered against it
        n_layers = len(self.model.model.layers)
        if not (0 <= self.layer < n_layers):
            raise ValueError(
                f"layer={self.layer} is out of range for {model_id}, which "
                f"has {n_layers} decoder layers (valid range: 0-{n_layers - 1})."
            )

        # storage for the hooked activation
        self._captured_activation = None
        self._hook_handle = None

        # Default system prompt used everywhere a chat-formatted generation
        # prompt is built. Forces short, single-word answers so generated
        # spans stay small and consistent across samples.
        self.default_system_prompt = (
            "Answer the question in a single word only. "
            "Do not use full sentences, explanations, or punctuation."
        )

    def _hook_fn(self, module, input, output):
        # decoder layers typically return a tuple; hidden_states is index 0
        hidden_states = output[0] if isinstance(output, tuple) else output
        self._captured_activation = hidden_states.detach()

    def _register_hook(self):
        target_layer = self.model.model.layers[self.layer]
        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)

    def _remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def get_layer_activations(self, text):
        """Run the prompt through Qwen and return the residual stream
        activations at self.layer, shape [seq_len, d_model]."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        self._register_hook()
        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            self._remove_hook()

        # [1, seq_len, d_model] -> [seq_len, d_model]
        activations = self._captured_activation.squeeze(0).to(torch.float32)
        return activations

    def generate_answer(
        self,
        question,
        max_new_tokens=8,
        use_chat_template=True,
        system_prompt=None,
        **gen_kwargs,
    ):
        """Generate an answer to `question` and return the answer text plus
        the full input_ids (prompt+answer) and the prompt length, so the
        caller can later isolate activations for just the generated span.

        By default the question is wrapped in the chat template with a
        system prompt instructing the model to answer in a single word,
        and max_new_tokens is kept small to match. Pass
        use_chat_template=False to fall back to feeding the raw question
        string with no instruction (old behavior).
        """
        if system_prompt is None:
            system_prompt = self.default_system_prompt

        if use_chat_template:
            prompt_text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = question

        prompt_inputs = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=not use_chat_template
        ).to(self.model.device)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **prompt_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                **gen_kwargs,
            )

        answer_ids = output_ids[0, prompt_len:]
        answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True)

        return {
            "answer_text": answer_text,
            "full_input_ids": output_ids,  # [1, prompt_len + answer_len]
            "prompt_len": prompt_len,
        }

    def get_feature_vector_for_ids(self, full_input_ids, start_idx, end_idx):
        """Run a forward pass over full_input_ids, grab layer activations for
        token positions [start_idx:end_idx) (e.g. the generated-answer span),
        mean-pool them, and return the raw residual-stream vector (size
        d_model) for that span — used directly as classifier input."""
        full_input_ids = full_input_ids.to(self.model.device)

        self._register_hook()
        try:
            with torch.no_grad():
                self.model(input_ids=full_input_ids)
        finally:
            self._remove_hook()

        activations = self._captured_activation.squeeze(0).to(torch.float32)  # [seq_len, d_model]
        span = activations[start_idx:end_idx]
        if span.shape[0] == 0:
            span = activations[-1:]  # fallback: last token if span is empty

        pooled = span.mean(dim=0)  # [d_model]
        return pooled

    def _find_answer_cutoff(self, answer_ids_row, stop_strings):
        """Walk the generated token ids one at a time and find the first
        point where the decoded text contains a stop string (e.g. the model
        starting a new "Question:" instead of stopping) or hits EOS. Returns
        the number of tokens to keep. If nothing triggers, keeps everything.
        This ensures the text we store and the tokens we pool for features
        always refer to the exact same span."""
        for i in range(len(answer_ids_row)):
            tok_id = answer_ids_row[i].item()
            if tok_id == self.tokenizer.eos_token_id:
                return i
            text_so_far = self.tokenizer.decode(answer_ids_row[: i + 1], skip_special_tokens=True)
            for stop in stop_strings:
                if stop in text_so_far:
                    return i + 1
        return len(answer_ids_row)

    def _format_chat_prompts(self, questions, system_prompt=None):
        """Wrap raw questions in Qwen's chat template — the format the
        Instruct model was actually trained to expect, so it reliably
        answers instead of continuing the raw string however it likes.

        Defaults to instructing the model to answer in a single word, so
        generated spans stay short and consistent across samples.
        """
        if system_prompt is None:
            system_prompt = self.default_system_prompt

        formatted = []
        for q in questions:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ]
            formatted.append(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        return formatted

    def batch_generate_and_extract_features(
        self, questions, max_new_tokens=8,
        stop_strings=("\nQuestion:", "\nQ:", "\n\n"), use_chat_template=True,
        conversation_histories=None, system_prompt=None, **gen_kwargs):
        """By default, generates a single-word answer per question (via the
        single-word system prompt baked into _format_chat_prompts /
        default_system_prompt) and max_new_tokens is kept small to match.
        Pass system_prompt explicitly to override, or use_chat_template=False
        to skip instruction-following entirely.

        Returns a list of dicts (one per question, same order as input):
        {"answer_text": str, "feature_vector": tensor[d_model]} — the
        feature_vector is the mean-pooled raw residual-stream activation
        over the generated-answer span, ready to feed into a probe."""
        if conversation_histories is not None:
            sys_prompt = system_prompt if system_prompt is not None else self.default_system_prompt
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": sys_prompt}] + hist + [{"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for hist, q in zip(conversation_histories, questions)
            ]
        else:
            prompts = (
                self._format_chat_prompts(questions, system_prompt=system_prompt)
                if use_chat_template
                else questions
            )

        prompt_inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=not use_chat_template
        ).to(self.model.device)
        max_prompt_len = prompt_inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **prompt_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                **gen_kwargs,
            )

        # answer tokens start right after the (left-padded, batch-aligned) prompt
        answer_ids_batch = output_ids[:, max_prompt_len:]

        cutoffs = [
            self._find_answer_cutoff(answer_ids_batch[i], stop_strings)
            for i in range(len(questions))
        ]
        answer_texts = [
            self.tokenizer.decode(answer_ids_batch[i, :cutoffs[i]], skip_special_tokens=True).strip()
            for i in range(len(questions))
        ]

        # build an attention mask covering [padding=0][real prompt=1][generated=1]
        full_attention_mask = torch.cat(
            [prompt_inputs["attention_mask"],
             torch.ones_like(answer_ids_batch)],
            dim=1,
        )

        self._register_hook()
        try:
            with torch.no_grad():
                self.model(input_ids=output_ids, attention_mask=full_attention_mask)
        finally:
            self._remove_hook()

        activations = self._captured_activation.to(torch.float32)  # [batch, seq_len, d_model]

        pooled_list = []
        for i in range(len(questions)):
            cutoff = max(cutoffs[i], 1)  # never pool an empty span
            span = activations[i, max_prompt_len: max_prompt_len + cutoff, :]
            pooled_list.append(span.mean(dim=0))
        pooled = torch.stack(pooled_list)  # [batch, d_model]

        return [
            {"answer_text": answer_texts[i], "feature_vector": pooled[i]}
            for i in range(len(questions))
        ]

    def batch_extract_features_for_qa_pairs(self, questions, answers):
        """No generation involved: feed (question, answer) as a completed
        chat turn — as if the model had said `answer` — and extract the
        mean-pooled raw residual-stream vector over just the answer-token
        span. Used for the "does the model's internal state look different
        when it's presented with a true vs. a hallucinated statement"
        framing, instead of generating an answer and guessing whether it
        hallucinated.

        Returns a list of dicts (one per question/answer pair, same order
        as input): {"feature_vector": tensor[d_model]}
        """
        # prefix = up to and including the assistant turn's start (no content yet)
        prefixes = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
            )
            for q in questions
        ]
        # full = prefix + the actual answer text, as the assistant's turn
        fulls = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                tokenize=False, add_generation_prompt=False,
            )
            for q, a in zip(questions, answers)
        ]

        # real (unpadded) prefix length per example — tells us where the
        # answer span starts once everything is batch-padded together
        prefix_lens = [
            len(self.tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prefixes
        ]

        full_inputs = self.tokenizer(
            fulls, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)
        seq_len = full_inputs["input_ids"].shape[1]
        real_lens = full_inputs["attention_mask"].sum(dim=1)  # real (non-pad) token count per row

        self._register_hook()
        try:
            with torch.no_grad():
                self.model(**full_inputs)
        finally:
            self._remove_hook()

        activations = self._captured_activation.to(torch.float32)  # [batch, seq_len, d_model]

        pooled_list = []
        for i in range(len(questions)):
            pad_amount = seq_len - real_lens[i].item()  # left-padding: real tokens are right-aligned
            start_idx = pad_amount + prefix_lens[i]
            end_idx = seq_len
            if start_idx >= end_idx:
                start_idx = end_idx - 1  # fallback: last token if answer span is empty
            span = activations[i, start_idx:end_idx, :]
            pooled_list.append(span.mean(dim=0))
        pooled = torch.stack(pooled_list)  # [batch, d_model]

        return [{"feature_vector": pooled[i]} for i in range(len(questions))]


# Backward-compatible alias — old code/imports that still say
# `from llama_features import LlamaSAEFeatureExtractor` keep working.
LlamaSAEFeatureExtractor = QwenResidualFeatureExtractor


if __name__ == "__main__":
    extractor = QwenResidualFeatureExtractor()
    acts = extractor.get_layer_activations("The cat sat on the mat.")
    print("Activation shape:", tuple(acts.shape))
    print("Mean-pooled feature vector shape:", tuple(acts.mean(dim=0).shape))
