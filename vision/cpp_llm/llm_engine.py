# vision/llm_engine.py
import os
import cv2
import json
import logging

import numpy as np
from PIL import Image
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class MultimodalInferencer:
    def __init__(
            self,
            model_path: str,
            n_gpu_layers: int = 0,      # 0 = CPU, -1 = GPU
            n_ctx: int = 4096,
            n_threads: int = 4,
            verbose: bool = False
    ):
        self.model_path = model_path
        logger.info(f"Loading LLM: {model_path} | GPU layers: {n_gpu_layers}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_batch=512,
            verbose=verbose,
            chat_format=None  # Автоопределение формата
        )

    def predict(
            self,
            crop_bgr: "np.ndarray",  # Кадр в формате cv2 (BGR)
            prompt_text: str,
            max_tokens: int = 256,
            temperature: float = 0.1,
            top_p: float = 0.9
    ) -> dict:
        """
        Принимает кроп в BGR, конвертирует в RGB для LLM,
        запускает инференс и возвращает распарсенный JSON.
        """
        try:
            # 1. Конвертация BGR -> RGB только для LLM
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(crop_rgb)

            # 2. Формирование мультимодального сообщения
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]

            # 3. Запуск через chat- Completion (стандарт для мультимодальных GGUF)
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=1.1
            )

            raw_text = response["choices"][0]["message"]["content"].strip()

            # 4. Очистка от markdown-обёрток (модели часто добавляют ```json)
            clean_json = raw_text.replace("```json", "").replace("```", "").strip()

            # 5. Безопасный парсинг
            parsed = json.loads(clean_json)
            parsed["uncertain"] = False
            return parsed

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed. Raw: {raw_text[:100]}...")
            return {"raw_text": raw_text, "uncertain": True, "error": str(e)}
        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            return {"uncertain": True, "error": str(e)}