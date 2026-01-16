# AGENTS.md — Project instructions for Codex

## 과제의 목적
- 이 과제의 목적은 MLCC 개발자들이 쓰는 플랫폼에서 행위를 도와주는 에이전트 개발이다.
- MLCC와 관련된 과제임을 명확히 인지하고 이에 대한 배경지식을 기반으로 프롬프트나 기타 용어 설정할것
- 워크플로우의 모든단계에서 수정 및 질문 요청에 대응할 수 있도록 구조 및 메모리를 설계해야한다.

## 0) Prime directive
- Before doing any work, load and follow these instructions.
- Prefer minimal changes that integrate with existing patterns.
- When unsure, inspect the codebase first (search, read nearby files) before implementing.

## 문서 업데이트 규칙
- `/api/chat` 워크플로우나 메모리 구조가 바뀌면 `docs/api-chat-workflow.md`를 같이 업데이트할 것

*** 1) 생각하는 규칙
- 사용자가 현재 코드에서 엣지케이스를 말하며 개선을 요청하면 딱 그 예시만 해결하려고 하지 말고, 그 예시가 왜 어떤 엣지케이스에서 비롯됐는지 파악하고, 그 근간을 해결할수 있는 해결책을 고민해야한다.
 - 코드를 수정하거나 추가 기능을 개발할 시 늘 사용자가 해당 과정에 대해서 변경요청을 할때에 잘 대응할 수 있을지 고려하여 메모리 구조를 짜야한다.
 - 코드는 늘 최대한 간결하고 컴팩트하게 작성한다. 불필요한 안전이나 검증절차는 최대한 지양한다.
 - 워크플로우 작성시 모델이 structured output format 을 만들어야 하는 상황이라면, openai 의 adk에 있는 structured output 기능을 이용한다.
 Reference: https://openai.github.io/openai-agents-python/


## 2) Skill-first policy (Codex Skills)
- Skills live under: ./.codex/skills/
- At the start of each task, scan available skills and decide whether one matches the task.
- If a skill matches, invoke it and follow it strictly.
- always consider to use advanced-evaluation, context-fundamentals, memory-systems, multi-agent-patterns, project-development, tool-design skills.

Default skill preferences (edit these to your actual skill names):
- Planning: $create-plan (use for non-trivial features/refactors)
- Repo conventions & code review checklist: $repo-review
- Testing & linting workflow: $test-and-lint
- Commit message drafting: $draft-commit-message

Notes:
- If a preferred skill exists but is not invoked automatically, explicitly invoke it (CLI: type `$skill-name`).
- If a rule in this file conflicts with a skill, follow the more specific instruction for the current directory/module.

# Coding Style Rules (Simple & Debuggable)

## Goal
Write code that a 1st-year developer can read and debug.
Prefer clarity and linear flow over abstraction, patterns, or defensive scaffolding.

---

## 1) Core Principles (must follow)
--코드의 각 단계마다 무엇을 하는지 한글 주석을 달아줘.
- **KISS**: simplest working solution first.
- **One obvious way**: avoid fancy patterns when a plain function works.
- **Linear control flow**: early returns are ok, but avoid deep nesting.
- **Small functions**: ~20–40 lines per function unless unavoidable.
- **Minimize indirection**: fewer layers, fewer wrappers, fewer helpers.

---

## 2) What to avoid (do NOT do)
### Over-engineering / abstraction
- No “framework-y” architecture unless explicitly requested.
- Avoid factories, registries, meta-programming, decorators for “cleverness”.
- Avoid creating extra modules/files just to “organize” unless necessary.

### Excessive defensive code
- Do **not** add broad try/except “just in case”.
- Do **not** add retry loops, circuit breakers, fallbacks unless user asked.
- Do **not** add complex validation layers unless required by an API contract.

### Hard-to-debug patterns
- No dynamic dispatch / reflection.
- No heavy generics / type wizardry.
- Avoid nested callbacks; prefer straightforward async/await or sync code.

---

## 3) What to do instead (preferred patterns)
### Errors & validation
- Validate only **inputs you truly need**.
- Fail fast with **clear error message** (raise once, don’t swallow).
- Catch exceptions **only** at:
  - external boundaries (API handler / CLI entrypoint)
  - where you can add meaningful context

### Logging
- Keep logging minimal and useful:
  - log request id / key parameters once
  - log success/failure once
- Do not spam logs inside tight loops.

### Naming & structure
- Use plain names: `fetch_user`, `parse_csv`, `save_result`
- Prefer 1 file per feature, not 1 file per tiny class.
- Prefer functions over classes unless stateful behavior is truly needed.

## 4) Explicit anti-pattern blacklist
Do not introduce:
- Repository/Service/UseCase layers without request
- “Base*” abstract classes
- Dependency injection containers
- Global exception swallowing (catch-all without re-raise)
- Magic constants without naming
- Implicit behavior (side effects in constructors/import time)