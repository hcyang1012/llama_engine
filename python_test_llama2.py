# The so file to be imported named `libllama_engine_pybind.so` is located in the build directory

# Import the llama module
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "build"))
import libllama_engine as llama

# Create a LlamaConfig object
config = llama.LlamaConfig()
config.checkpoint_path = "stories15M.bin"
config.tokenizer_path = "tokenizer.bin"
config.device_type = llama.DeviceType.CUDA

# Create a Llama2 object using the config
llama2 = llama.Llama2FP32(config)

# Create a RunConfig object
run_config = llama.RunConfig()
run_config.temperature = 0.7
run_config.topp = 0.9
run_config.rng_seed = 42


# Generate text
text = llama2.Generate("Once upon a time", 128, run_config, False)
print(text)
