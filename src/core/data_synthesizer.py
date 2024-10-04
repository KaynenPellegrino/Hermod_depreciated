class DataSynthesizer:
    def __init__(self, gpt_client):
        """Initialize the synthesizer with GPTClient."""
        self.gpt_client = gpt_client

    def synthesize_data(self, task_description):
        """Generate synthetic data based on the task description."""
        prompt = f"Generate synthetic data for the task: {task_description}"
        synthetic_data = self.gpt_client.generate_code(prompt)
        return synthetic_data

    def generate_synthetic_code(self, project_name):
        """Generate synthetic code using GPT."""
        prompt = f"Generate example buggy code for a project named {project_name}."
        return self.gpt_client.generate_code(prompt)
