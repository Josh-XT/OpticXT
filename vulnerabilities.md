# Security Vulnerabilities for Josh-XT/OpticXT

**Repository:** https://github.com/Josh-XT/OpticXT
**Total Alerts:** 1
**Generated:** 2026-04-19T15:12:29.116635+00:00

---

## 1. [LOW] ✅ RESOLVED - Rand is unsound with a custom logger using rand::rng()

- **Type:** Dependabot
- **Severity:** low
- **Package:** rand
- **Ecosystem:** rust
- **Vulnerable Range:** >= 0.7.0, < 0.9.3
- **Patched Version:** 0.9.3
- **Manifest File:** Cargo.toml
- **Scope:** runtime
- **Dependency Type:** unknown
- **GHSA:** GHSA-cq8v-f236-94qc
- **CWEs:** CWE-20: Improper Input Validation
- **Description:** It has been reported (by @lopopolo) that the `rand` library is [unsound](https://rust-lang.github.io/unsafe-code-guidelines/glossary.html#soundness-of-code--of-a-library) (i.e. that safe code using the public API can cause Undefined Behaviour) when all the following conditions are met:

- The `log` and `thread_rng` features are enabled
- A [custom logger](https://docs.rs/log/latest/log/#implementing-a-logger) is defined
- The custom logger accesses `rand::rng()` (previously `rand::thread_rng()`)
- **References:**
  - https://github.com/rust-random/rand/pull/1763
  - https://rustsec.org/advisories/RUSTSEC-2026-0097.html
  - https://github.com/advisories/GHSA-cq8v-f236-94qc
- **URL:** https://github.com/Josh-XT/OpticXT/security/dependabot/1

---

IMPORTANT: Do NOT commit directly to the main branch. Create a new branch named `fix/security-vulnerabilities` from the default branch, make all changes there, then open a pull request back to the default branch.

Please fix all of the above vulnerabilities. For dependency vulnerabilities, update the affected packages to their patched versions. For code scanning issues, fix the code at the specified locations. For security advisories, review the reported vulnerability and implement the necessary code fixes to address them.