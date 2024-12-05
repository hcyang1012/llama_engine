# 🚀 LLAMA Engine : A Simple C++/CUDA Library to Run LLAMA2 and LLAMA3

LLAMA Engine is a simple C++/CUDA library designed to run LLAMA2 and LLAMA3. 

## 📖 Project Description
This project is a fork of the following projects:
- [LLAMA2 implementation](https://github.com/karpathy/llama2.c)
- [LLAMA3 implementation](https://github.com/jameswdelancey/llama3.c)
- [CUDA kernel](https://github.com/likejazz/llama3.cuda) - The CUDA code has been copied as it is.

## ✨ Features

- **🚀 C++/CUDA Integration**: Seamlessly integrates C++ and CUDA for high-performance computations.
- **🦙 Support for LLAMA2 and LLAMA3**: Specifically developed for running LLAMA2 and LLAMA3 models.
- **👌 Easy to Use**: Simple API for quick setup and execution.

## 🛠️ Example Installation
To install LLAMA Engine, follow these steps:

> **Note**: This is a header-only project, so you can easily adopt it by including the `include` directory in your project's build.

1. Clone the repository:
    ```sh
    git clone git@github.com:hcyang1012/llama_engine.git
    ```
2. Include the `include` directory in your project's build system.

3. Build the project:
    ```sh
    cd build
    cmake .. -DUSE_CUDA=OFF # Build without CUDA 
    cmake .. -DUSE_CUDA=ON  # Build with CUDA
    make
    ```

## 📦 Model Download

Convert Meta's model to the required binary format. Refer to these projects:

- LLAMA2: [llama2.c](https://github.com/karpathy/llama2.c)
- LLAMA3: [llama3.c](https://github.com/jameswdelancey/llama3.c)

Tested models:
- LLAMA2: stories15M model from llama2.c.
- LLAMA3: llama3.1 8B-Instruct

## 🚀 Usage

Here is a basic example of how to use LLAMA Engine:
```cpp
#include <llama.hpp>
#include <iostream>

int main() {
    // Set up the configuration for LLAMA2
    llama::LlamaConfig llama2_config = {
        .checkpoint_path = "path/to/llama2/checkpoint",
        .tokenizer_path = "path/to/llama2/tokenizer.bin",
        .device_type = llama::DeviceType::CPU // or llama::DeviceType::CUDA
    };
    llama::Llama2<float> llama2(llama2_config);

    // Generate text using LLAMA2
    const char* prompt = "Hello, how are you?";
    llama::RunConfig run_config = {
        .temperature = 1.0f,
        .topp = 0.9f,
        .rng_seed = 42
    };
    llama2.Generate(prompt, 256, run_config);

    // Set up the configuration for LLAMA3
    llama::LlamaConfig llama3_config = {
        .checkpoint_path = "path/to/llama3/checkpoint",
        .tokenizer_path = "path/to/llama3/tokenizer_llama3.bin",
        .device_type = llama::DeviceType::CPU // or llama::DeviceType::CUDA
    };
    llama::Llama3<float> llama3(llama3_config);

    // Generate text using LLAMA3
    llama3.Generate(prompt, 256, run_config);

    return 0;
}
```

## Acknowledgements
Special thanks to the authors of the original projects:

- karpathy
- jameswdelancey
- likejazz

## Contact
For any questions or suggestions, feel free to open an issue or contact us at heecheol.yang@outlook.com

Happy coding! 🎉 ```