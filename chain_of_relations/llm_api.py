#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   llm_api.py
@Time    :   2026/02/11 08:35:53
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""


import os
import time
import logging
from openai import OpenAI


class LLMAPI(object):

    def __init__(self, model_name=None, model_type=None):
        self.model_name = model_name or model_type or os.getenv("MODEL_NAME")
        
        # 从环境变量读取配置
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
        self.timeout = float(os.getenv("OPENAI_TIMEOUT", "30"))
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
        self.retry_interval = float(os.getenv("OPENAI_RETRY_INTERVAL", "1"))

        if not self.model_name:
            raise ValueError("MODEL_NAME not found in environment variables and no model_name/model_type provided")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # The wrapper owns retry behavior, so disable the SDK's additional
        # automatic retries.  Otherwise one logical attempt can become three
        # HTTP attempts inside the SDK before the wrapper retries again.
        effective_base_url = self.base_url or "https://api.openai.com/v1"
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": effective_base_url,
            "timeout": self.timeout,
            "max_retries": 0,
        }
        self.client = OpenAI(**client_kwargs)
        logging.info(
            f"Initialized OpenAI client with model: {self.model_name}, "
            f"base_url: {effective_base_url}, timeout: {self.timeout}s"
        )

    def generate(self, user_prompt, temperature=0.01, max_tokens=256, system_prompt=None):
        """
        统一使用OpenAI标准接口进行调用
        支持所有兼容OpenAI API的服务商：OpenAI、Pumpkin、SiliconFlow、DeepSeek等
        """
        logging.info(f"Requesting model: {self.model_name}")

        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        last_error = None
        modern_model = str(self.model_name).startswith(("gpt-5", "gpt-6"))
        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "timeout": self.timeout,
        }
        # Current reasoning models reject max_tokens and require the renamed
        # max_completion_tokens field.  Keep the public Python argument named
        # max_tokens so existing callers remain unchanged.
        if modern_model:
            # Current reasoning models use their fixed/default sampling
            # behavior and reject non-default temperature values (as well as
            # legacy sampling controls).  Keep only supported controls.
            request_kwargs["max_completion_tokens"] = max_tokens
            request_kwargs["reasoning_effort"] = os.getenv("OPENAI_REASONING_EFFORT", "low")
        else:
            request_kwargs.update({
                "temperature": temperature,
                "max_tokens": max_tokens,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            })

        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**request_kwargs)

                result = None
                if getattr(response, "choices", None):
                    message = getattr(response.choices[0], "message", None)
                    if message is not None:
                        result = getattr(message, "content", None)

                usage_obj = getattr(response, "usage", None)
                if usage_obj:
                    usage = {
                        "input_tokens": getattr(usage_obj, "prompt_tokens", 0),
                        "output_tokens": getattr(usage_obj, "completion_tokens", 0),
                        "total_tokens": getattr(usage_obj, "total_tokens", 0),
                    }

                if result:
                    return result, usage

                last_error = RuntimeError("empty_model_response")
                logging.warning(
                    f"Model response has empty content, retry {attempt}/{self.max_retries}"
                )
            except Exception as e:
                last_error = e
                cause = getattr(e, "__cause__", None)
                cause_text = f"; cause={cause!r}" if cause else ""
                logging.error(
                    f"API error ({type(e).__name__}: {e}{cause_text}), "
                    f"retry {attempt}/{self.max_retries}"
                )

            if attempt < self.max_retries:
                time.sleep(self.retry_interval)

        logging.error(f"Failed to get response after {self.max_retries} retries")
        usage["error"] = str(last_error) if last_error else "unknown_error"
        return None, usage
