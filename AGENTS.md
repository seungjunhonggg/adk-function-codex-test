# AGENTS.md — Project instructions for Codex

## 과제의 목적
- 이 과제의 목적은 MLCC 개발자들이 쓰는 플랫폼에서 행위를 도와주는 에이전트 개발이다.
- MLCC와 관련된 과제임을 명확히 인지하고 이에 대한 배경지식을 기반으로 프롬프트나 기타 용어 설정할것

## 0) Prime directive
- Before doing any work, load and follow these instructions.
- Prefer minimal changes that integrate with existing patterns.
- When unsure, inspect the codebase first (search, read nearby files) before implementing.

## 문서 업데이트 규칙
- `/api/chat` 워크플로우나 메모리 구조가 바뀌면 `docs/api-chat-workflow.md`를 같이 업데이트할 것
- `backend/reference_lot_rules.json` 수정/추가 시 `docs/reference-lot-rules.yaml`에도 동일 구조로 주석을 업데이트할 것

*** 1) 생각하는 규칙
- 사용자가 현재 코드에서 엣지케이스를 말하며 개선을 요청하면 딱 그 예시만 해결하려고 하지 말고, 그 예시가 왜 어떤 엣지케이스에서 비롯됐는지 파악하고, 그 근간을 해결할수 있는 해결책을 고민해
 - 코드를 수정하거나 추가 기능을 개발할 시 늘 사용자가 해당 과정에 대해서 변경요ㅓㅇ을 할때에 잘 대응할 수 있을지 고려하여 메모리 구조를 짜야한다.
<!-- ## 1) Always consider OpenAI Agents SDK first
- When implementing anything "agentic" (multi-agent orchestration, handoffs, tool calling, guardrails, tracing),
  always check whether the OpenAI Agents SDK already provides a supported primitive/pattern.
- Reference: https://openai.github.io/openai-agents-python/

Practical checklist:
- Can this be modeled as: Agent + tools + handoffs + Runner?
- Do we need structured tool schemas, guardrails, or tracing?
- 안정적인 production을 구축하고자 할때, openai adk보다 코드로 워크플로우를 구성하는게 더 안정적이라면 그렇게 실행해도 된다.
- If Codex CLI automation is involved, consider MCP-based orchestration patterns. -->

## 2) Skill-first policy (Codex Skills)
- Skills live under: ./.codex/skills/
- At the start of each task, scan available skills and decide whether one matches the task.
- If a skill matches, invoke it and follow it strictly.

Default skill preferences (edit these to your actual skill names):
- Planning: $create-plan (use for non-trivial features/refactors)
- Repo conventions & code review checklist: $repo-review
- Testing & linting workflow: $test-and-lint
- Commit message drafting: $draft-commit-message

Notes:
- If a preferred skill exists but is not invoked automatically, explicitly invoke it (CLI: type `$skill-name`).
- If a rule in this file conflicts with a skill, follow the more specific instruction for the current directory/module.
