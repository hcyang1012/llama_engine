#include <iostream>
#include <run_state.hpp>
#include <transformer.hpp>
int main(int, char **) {
  std::cout << "Hello, from llama2_cpp_dummy!\n";
  llama2::Transformer<float> transformer("config.json", "chkpt.pt");
}
