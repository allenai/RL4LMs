from rl4lms.data_pools.text_generation_pool import TextGenPool, Sample


class TestTextGenPool(TextGenPool):
    @classmethod
    def prepare(cls, split: str, prompt: str, n_samples=100):
        samples = [Sample(id=ix,
                          prompt_or_input_text=prompt,  # a dummy prompt
                          references=[]
                          ) for ix in range(n_samples)]
        pool_instance = cls(samples)
        return pool_instance
