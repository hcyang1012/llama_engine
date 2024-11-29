

// C System-Headers

// C++ System-Headers
#include <cstring>
#include <iostream>

// Project Headers
#include <dtypes.h>

#include <op.hpp>
#include <transformer.hpp>

#if defined(USE_LLAMA2)
#include "references/reference_llama2.cpp"
#endif
// Third-party Headers

void error_usage(const std::string &program_name) {
  std::cerr << "Usage: " << program_name << " [options] checkpoint_path\n";
  std::cerr << "Options:\n";
  std::cerr << "  -t <temperature>    Set the temperature (default: 1.0)\n";
  std::cerr << "  -p <topp>           Set the top-p value (default: 0.9)\n";
  std::cerr << "  -s <rng_seed>       Set the random number generator seed "
               "(default: current time)\n";
  std::cerr << "  -n <steps>          Set the number of steps to run for "
               "(default: 256)\n";
  std::cerr << "  -i <prompt>         Set the prompt string\n";
  std::cerr << "  -m <mode>           Set the mode (default: generate)\n";
  std::cerr << "  -y <system_prompt>  Set the system prompt\n";
  std::cerr << "  -z <tokenizer_path> Set the tokenizer path (default: "
               "tokenizer.bin)\n";
  std::exit(1);
}

int main(int argc, char *argv[]) {
  // default parameters
  std::string checkpoint_path;
  std::string tokenizer_path("tokenizer.bin");
  float temperature =
      1.0f;  // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp =
      0.9f;  // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  llama::llama_uint32_t steps = 256;  // number of steps to run for
  char *prompt = NULL;                // prompt string
  unsigned long long rng_seed = 0;    // seed rng with time by default
  std::string mode("generate");       // mode to run in
  char *system_prompt =
      NULL;  // the (optional) system prompt to use in chat mode

  // poor man's C argparse so we can override the defaults above from the
  // command line
  if (argc >= 2) {
    checkpoint_path = argv[1];
  } else {
    error_usage(argv[0]);
  }
  for (int i = 2; i < argc; i += 2) {
    // do some basic validation
    if (i + 1 >= argc) {
      error_usage(argv[0]);
    }  // must have arg after flag
    if (argv[i][0] != '-') {
      error_usage(argv[0]);
    }  // must start with dash
    if (strlen(argv[i]) != 2) {
      error_usage(argv[0]);
    }  // must be -x (one dash, one letter)
    // read in the args
    if (argv[i][1] == 't') {
      temperature = atof(argv[i + 1]);
    } else if (argv[i][1] == 'p') {
      topp = atof(argv[i + 1]);
    } else if (argv[i][1] == 's') {
      rng_seed = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'n') {
      steps = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'i') {
      prompt = argv[i + 1];
    } else if (argv[i][1] == 'm') {
      mode = argv[i + 1];
    } else if (argv[i][1] == 'y') {
      system_prompt = argv[i + 1];
    } else if (argv[i][1] == 'z') {
      tokenizer_path = argv[i + 1];
    } else {
      error_usage(argv[0]);
    }
  }

  // parameter validation/overrides
  if (rng_seed <= 0) {
    rng_seed = (unsigned int)time(NULL);
  }
  if (temperature < 0.0) {
    temperature = 0.0;
  }
  if (topp < 0.0 || 1.0 < topp) {
    topp = 0.9;
  }
  if (steps < 0) {
    steps = 0;
  }
  const std::string kDefaultPrompt = "hello, how are you?";
  if (prompt == NULL) {
    prompt = const_cast<char *>(kDefaultPrompt.c_str());
  }

  const llama::Transformer<float>::RunConfig run_config = {temperature, topp,
                                                           rng_seed};

  const llama::DeviceType device_type = llama::DeviceType::CPU;
  auto op_set = llama::CreateOpSet(device_type);
  const auto special_tokens = llama::SpecialTokensLlama2();
  // build the Transformer via the model .bin file
  llama::Transformer<float> transformer(checkpoint_path, run_config, *op_set,
                                        special_tokens);

  if (steps == 0) {
    steps = transformer.GetConfig().SeqLen();
  }

  steps = std::min(steps, transformer.GetConfig().SeqLen());

  llama::Tokenizer<float> tokenizer(tokenizer_path,
                                    transformer.GetConfig().VocabSize());

  if (mode == "generate") {
    {
      reference_llama2::Transformer ref_transformer;
      reference_llama2::build_transformer(&ref_transformer,
                                          checkpoint_path.c_str());

      reference_llama2::Tokenizer ref_tokenizer;
      reference_llama2::build_tokenizer(&ref_tokenizer, tokenizer_path.c_str(),
                                        ref_transformer.config.vocab_size);

      reference_llama2::Sampler sampler;
      reference_llama2::build_sampler(&sampler,
                                      ref_transformer.config.vocab_size,
                                      temperature, topp, rng_seed);

      std::cout << "Reference Generation:" << std::endl;
      reference_llama2::generate(&ref_transformer, &ref_tokenizer, &sampler,
                                 prompt, steps);
    }
    std::cout << std::endl;
    {
      std::cout << "My Generation:" << std::endl;
      transformer.Generate(tokenizer, prompt, steps);
    }

  } else if (mode == "chat") {
    // transformer.Chat(tokenizer,
    // sampler, steps, prompt,
    // system_prompt);
  } else {
    std::cerr << "Unknown mode: " << mode << std::endl;
    error_usage(argv[0]);
  }
  return 0;
}
