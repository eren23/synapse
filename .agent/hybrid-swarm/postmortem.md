# Swarm Post-Mortem Report

## Summary
- **Outcome**: completed
- **Tasks**: 18/18 completed
- **Success rate**: 100%
- **Cost**: $0.0000
- **Duration**: 13569.6s

## Decomposition Quality
- **Score**: 0.80/1.00
- **Parallel efficiency**: 24%
- **Issues**: 130

## Execution
- **Critical path**: task-1 → task-5 → task-6 → task-9 → task-10 → task-11 → task-12 → task-13 → task-17 → task-18

## Failures
### Root Causes
- **task-3** (agent_error): blocked 1 tasks, wasted $0.0000
- **task-4** (agent_error): blocked 1 tasks, wasted $0.0000
- **task-8** (agent_error): blocked 1 tasks, wasted $0.0000
- **task-6** (agent_error): blocked 1 tasks, wasted $0.0000
- **task-9** (agent_error): blocked 1 tasks, wasted $0.0000
- **task-10** (agent_error): blocked 1 tasks, wasted $0.0000
- **task-13** (agent_error): blocked 1 tasks, wasted $0.0000
- **task-16** (agent_error): blocked 1 tasks, wasted $0.0000

**Total wasted cost**: $0.0000

## Robustness Events
- Rate limits: 0
- Concurrency adjustments: 26
- Budget warnings: 0

## Recommendations
- Low parallel efficiency — review dependency graph for unnecessary sequential constraints
