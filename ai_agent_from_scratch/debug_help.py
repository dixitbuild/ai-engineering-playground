
from langchain_core.callbacks import BaseCallbackHandler

class DebugLLMCallback(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)

                if message:
                    print(
                        "\nFinish reason:",
                        message.response_metadata.get("finish_reason")
                    )
                    print(
                        "Response metadata:",
                        message.response_metadata
                    )
                    print(
                        "Usage:",
                        message.usage_metadata
                    )
