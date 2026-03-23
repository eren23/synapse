# Swarm Post-Mortem Report

## Summary
- **Outcome**: completed
- **Tasks**: 15/15 completed
- **Success rate**: 100%
- **Cost**: $0.0000
- **Duration**: 10548.6s

## Decomposition Quality
- **Score**: 0.15/1.00
- **Parallel efficiency**: 50%
- **Issues**: 76

## Execution
- **Critical path**: task-1 → task-5 → task-7 → task-12 → task-13 → task-14 → task-15

## Failures
### Root Causes
- **task-8** (agent_error): blocked 1 tasks, wasted $0.0000
- **task-15** (agent_error): blocked 1 tasks, wasted $0.0000

**Total wasted cost**: $0.0000

## Robustness Events
- Rate limits: 0
- Concurrency adjustments: 8
- Budget warnings: 0

## Recommendations
- Low decomposition quality — review task granularity and dependency edges
