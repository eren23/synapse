# Swarm Post-Mortem Report

## Summary
- **Outcome**: completed
- **Tasks**: 15/15 completed
- **Success rate**: 100%
- **Cost**: $0.0000
- **Duration**: 8175.1s

## Decomposition Quality
- **Score**: 0.20/1.00
- **Parallel efficiency**: 50%
- **Issues**: 24

## Execution
- **Critical path**: task-1 → task-4 → task-5 → task-6 → task-13 → task-14 → task-15

## Failures
### Root Causes
- **task-12** (agent_error): blocked 1 tasks, wasted $0.0000
- **task-14** (agent_error): blocked 1 tasks, wasted $0.0000

**Total wasted cost**: $0.0000

## Robustness Events
- Rate limits: 0
- Concurrency adjustments: 8
- Budget warnings: 0

## Recommendations
- Low decomposition quality — review task granularity and dependency edges
