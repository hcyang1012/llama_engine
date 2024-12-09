# The so file to be imported named `libllama_engine_pybind.so` is located in the build directory

# Import the llama module
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "build"))
import libllama_engine as llama

# Create a LlamaConfig object
config = llama.LlamaConfig()
config.checkpoint_path = "llama3_1-8b-instruct.bin"
config.tokenizer_path = "llama3_tokenizer.bin"
config.device_type = llama.DeviceType.CUDA

# Create a Llama3 object using the config
llama3 = llama.Llama3FP32(config)

# Create a RunConfig object
run_config = llama.RunConfig()
run_config.temperature = 0.7
run_config.topp = 0.9
run_config.rng_seed = 42


# Generate text
text = llama3.Generate("Once upon a time", 1024, run_config, False)
print(text)
