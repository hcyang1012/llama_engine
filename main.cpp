#include <cstring>
#include <iostream>
#include <run_state.hpp>
#include <transformer.hpp>

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
  std::exit(1);
}

int main(int argc, char *argv[]) {
  // default parameters
  std::string checkpoint_path;
  std::string tokenizer_path("tokenizer.bin");
  float temperature =
      1.0f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp =
      0.9f; // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int steps = 256;                 // number of steps to run for
  char *prompt = NULL;             // prompt string
  unsigned long long rng_seed = 0; // seed rng with time by default
  std::string mode("generate");    // mode to run in
  char *system_prompt =
      NULL; // the (optional) system prompt to use in chat mode

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
    } // must have arg after flag
    if (argv[i][0] != '-') {
      error_usage(argv[0]);
    } // must start with dash
    if (strlen(argv[i]) != 2) {
      error_usage(argv[0]);
    } // must be -x (one dash, one letter)
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
    } else {
      error_usage(argv[0]);
    }
  }

  // parameter validation/overrides
  if (rng_seed <= 0)
    rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0)
    temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp)
    topp = 0.9;
  if (steps < 0)
    steps = 0;

  // build the Transformer via the model .bin file
  llama2::Transformer<float> transformer(checkpoint_path);
  return 0;
}
