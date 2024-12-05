#pragma once
#include <chrono>
#include <fstream>
#include <json/json.hpp>
#include <list>
#include <string>
class Tracer {
 public:
  class Work {
   public:
    Work(const std::string& name, const size_t tid = 0) : name(name), tid(tid) {
      start = std::chrono::high_resolution_clock::now();
    }

    void stop() { end = std::chrono::high_resolution_clock::now(); }

    nlohmann::json to_json() {
      nlohmann::json event;
      event["name"] = name;
      event["ph"] = "X";
      event["ts"] = std::chrono::duration_cast<std::chrono::microseconds>(
                        start.time_since_epoch())
                        .count();
      event["dur"] =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
      event["pid"] = 0;
      event["tid"] = tid;
      return event;
    }

   private:
    const std::string name;
    const size_t tid;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
  };
  static Tracer& GetInstance() {
    static Tracer instance;
    return instance;
  }
  Tracer() {}
  ~Tracer() {}

  Work& start_work(const std::string& name, const size_t tid = 0) {
    works.emplace_back(name, tid);
    return works.back();
  }

  void to_json(std::string filename) {
    nlohmann::json json;
    auto trace_events = nlohmann::json::array();
    for (auto& work : works) {
      trace_events.push_back(work.to_json());
    }
    json["traceEvents"] = trace_events;
    json["displayTimeUnit"] = "ns";

    std::ofstream out(filename);
    out << json.dump(2);
  }

 private:
  std::list<Work> works;
};