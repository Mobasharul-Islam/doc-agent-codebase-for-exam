"""Run tasks.jsonl through the agent and score."""
import json
import sys

from doc_agent import config
from doc_agent.eval import ablation
from doc_agent.mlops import tracking


def main() -> None:
    cfg = config.load()
    tasks = ablation.load_tasks()
    if not tasks:
        sys.exit("grading_kit/tasks.jsonl has no filled-in questions")
    tracking.init_run(cfg, ["eval"])
    res = ablation.evaluate(cfg, tasks)
    tracking.log(res)
    print(json.dumps(res, indent=2))
    if "--ablation" in sys.argv:
        print(json.dumps(ablation.run(cfg), indent=2))


if __name__ == "__main__":
    main()
