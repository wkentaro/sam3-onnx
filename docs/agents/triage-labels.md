# Triage labels

`/triage` and the PR skills read this file to turn role names into the label
strings this repo uses.

| Role              | Issue label                     | PR label                    |
| ----------------- | ------------------------------- | --------------------------- |
| `bug`             | `type:bug`                      | none                        |
| `enhancement`     | `type:feature` or `type:task`   | none                        |
| `needs-triage`    | `needs-triage`                  | `recommend-triage`          |
| `needs-info`      | `needs-info`                    | `recommend-revise`          |
| `ready-for-agent` | `ready-for-agent`               | no verdict label, non-draft |
| `ready-for-human` | `ready-for-human`               | `recommend-merge`           |
| `wontfix`         | `wontfix`                       | `recommend-close`           |

`enhancement` maps to `type:task` when the request is maintenance, docs, or
refactor work. A PR gets no type label because its Conventional Commits title
already says it.

`maintainer-approved` has no role. It goes on a PR only when the maintainer
says so.

## Rules

- A triaged issue carries exactly one `type:` label and one triage state. An
  unlabeled issue is untriaged; `needs-triage` means under evaluation.
- A non-draft PR with no verdict is the agent's to finalize. The draft flag is
  the "still being built" state; there is no label for it.
- The agent emits at most one `recommend-*` verdict per head.
  `recommend-revise` hands the PR back to its author; the other three hand it
  to the maintainer.
- Verdicts are recommendations. The agent never merges and never closes. Never
  rename `recommend-merge` to `ready-to-merge`: merge bots watch that string.
- `maintainer-approved` is set only on explicit maintainer direction and may
  coexist with a `recommend-*` label.
- A new push makes any verdict stale. The authority that set it clears and
  renews it.
